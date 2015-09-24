// This file is a part of Julia. License is MIT: http://julialang.org/license

/*
  task.c
  lightweight processes (symmetric coroutines)
*/

#if defined(__APPLE__) && defined(HAVE_UCONTEXT)
// need this to get the real definition of ucontext_t
#define _XOPEN_SOURCE
#endif

#ifdef _FORTIFY_SOURCE
// disable __longjmp_chk validation so that we can jump between stacks
#pragma push_macro("_FORTIFY_SOURCE")
#undef _FORTIFY_SOURCE
#include <setjmp.h>
#pragma pop_macro("_FORTIFY_SOURCE")
#endif

#include "platform.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <errno.h>
#include <inttypes.h>
#include "julia.h"
#include "julia_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_OS_WINDOWS_)
#include <winbase.h>
#include <malloc.h>
#include <dbghelp.h>
volatile int jl_in_stackwalk = 0;
#else
#include <unistd.h>
#include <sys/mman.h> // for mprotect
#include <dlfcn.h>   // for dladdr
// This gives unwind only local unwinding options ==> faster code
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

//  Options for this file (in order of preference):
// HAVE_FIBER (aka _OS_WINDOWS_)
// HAVE_ASM
// HAVE_UCONTEXT
// HAVE_UNW_CONTEXT
// HAVE_SIGALTSTACK

#if !defined(HAVE_FIBER) && \
    !defined(HAVE_UCONTEXT) && \
    !defined(HAVE_ASM) && \
    !defined(HAVE_UNW_CONTEXT) && \
    !defined(HAVE_SIGALTSTACK)
#if defined(_OS_WINDOWS_)
#define HAVE_FIBER
#elif (defined(_CPU_X86_64_) || defined(_CPU_X86_)) && !defined(_COMPILER_MICROSOFT_)
#define HAVE_ASM
#elif defined(_OS_DARWIN_)
#define HAVE_UNW_CONTEXT
#elif defined(_OS_LINUX_)
#define HAVE_UCONTEXT
#else
#define HAVE_UNW_CONTEXT
#endif
#endif

#ifdef HAVE_UCONTEXT
#include <ucontext.h>
#endif

/* This probing code is derived from Douglas Jones' user thread library */

/* true if stack grows up, false if down */
static int _stack_grows_up;

static void _infer_direction_from(int *first_addr)
{
    int second;
    _stack_grows_up = (first_addr < &second);
}

static void _infer_stack_direction(void)
{
    int first;
    _infer_direction_from(&first);
}

static void _probe_arch(void)
{
    _infer_stack_direction();
}

/* end probing code */

#ifndef _OS_WINDOWS_
#ifdef _OS_DARWIN_
#define MAP_ANONYMOUS MAP_ANON
#endif
static void *malloc_stack(size_t bufsz)
{
    void* stk = mmap(0, bufsz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (stk == MAP_FAILED)
        jl_throw(jl_memory_exception);
#if !defined(HAVE_UCONTEXT) && !defined(HAVE_SIGALTSTACK)
    // setup a guard page to detect stack overflow
    if (mprotect(stk, jl_page_size, PROT_NONE) == -1) {
        munmap(stk, bufsz);
        jl_errorf("mprotect: %s", strerror(errno));
    }
#endif
    return stk;
}

static void free_stack(void *stkbuf, size_t bufsz)
{
    munmap(stkbuf, bufsz);
}
#endif

#ifndef MINSIGSTKSZ
#define MINSIGSTKSZ 131072 // 128k
#endif

static jl_sym_t *done_sym;
static jl_sym_t *failed_sym;
static jl_sym_t *runnable_sym;

extern size_t jl_page_size;
jl_datatype_t *jl_task_type;
DLLEXPORT JL_THREAD jl_task_t * volatile jl_current_task;
JL_THREAD jl_task_t *jl_root_task;
DLLEXPORT JL_THREAD jl_value_t *jl_exception_in_transit;
DLLEXPORT JL_THREAD jl_gcframe_t *jl_pgcstack = NULL;
static void jl_new_fiber(jl_task_t *t, size_t ssize);

#ifdef HAVE_UNW_CONTEXT
static unw_context_t jl_base_uctx;
static unw_cursor_t jl_basecursor;
#endif

#ifdef HAVE_UCONTEXT
static ucontext_t jl_base_uctx;
#endif

JL_THREAD void *jl_stackbase; // only needed by COPY_STACKS, but simplifies the code to have it defined always
#ifdef COPY_STACKS
static JL_THREAD jl_jmp_buf jl_basectx;
#ifdef _OS_WINDOWS_
static JL_THREAD LPVOID jl_basefiber;
#endif

static void NOINLINE save_stack(jl_task_t *t)
{
    if (t->state == done_sym || t->state == failed_sym)
        return;
    volatile char *_x;
    size_t nb = (char*)jl_stackbase - (char*)&_x;
    char *buf;
    if (t->stkbuf == NULL || ((size_t*)t->stkbuf)[-1] < nb) {
        buf = (char*)realloc(t->stkbuf ? t->stkbuf - sizeof(size_t) : NULL,
                nb + sizeof(size_t));
        if (buf == NULL)
            jl_throw(jl_memory_exception);
        t->stkbuf = buf + sizeof(size_t);
        ((size_t*)t->stkbuf)[-1] = nb;
    }
    t->ssize = nb;
    memcpy(t->stkbuf, (char*)&_x, nb);
    // this task's stack could have been modified after
    // it was marked by an incremental collection
    // move the barrier back instead of walking it again here
    jl_gc_wb_back(t);
}

void NOINLINE NORETURN restore_stack(jl_task_t *t, char *p)
{
    char *_x = (char*)jl_stackbase - t->ssize;
    if (!p) {
        p = _x;
        if ((char*)&_x > _x) {
            p = (char*)alloca((char*)&_x - _x);
        }
        restore_stack(t, p); // pass p as a parameter so that the compiler can't avoid the alloca
    }
    assert(t->stkbuf != NULL);
    memcpy(_x, t->stkbuf, t->ssize);
    jl_longjmp(t->ctx, 1);
    abort();
}
#endif

static jl_function_t *task_done_hook_func=NULL;

static void NORETURN finish_task(jl_task_t *t, jl_value_t *resultval)
{
    if (t->exception != jl_nothing)
        t->state = failed_sym;
    else
        t->state = done_sym;
    t->result = resultval;
    jl_gc_wb(t, t->result);
#ifdef COPY_STACKS
    // early free of stkbuf, if we aren't running on it right now
    void *stkbuf = t->stkbuf;
    if (t->copy_stack && stkbuf != NULL && stkbuf != (void*)(intptr_t*)-1) {
        t->stkbuf = (void*)(intptr_t)-1;
        free(stkbuf - sizeof(size_t));
    }
#endif
    if (task_done_hook_func == NULL) {
        task_done_hook_func = (jl_function_t*)jl_get_global(jl_base_module,
                                                            jl_symbol("task_done_hook"));
    }
    if (task_done_hook_func != NULL) {
        jl_apply(task_done_hook_func, (jl_value_t**)&t, 1);
    }
    abort();
}

static void throw_if_exception_set(jl_task_t *t)
{
    if (t->exception != NULL && t->exception != jl_nothing) {
        jl_value_t *exc = t->exception;
        t->exception = jl_nothing;
        jl_throw(exc);
    }
}

static void NOINLINE NORETURN start_task()
{
    // this runs the first time we switch to a task
    jl_task_t *t = jl_current_task;
    throw_if_exception_set(t);
    jl_value_t *res = jl_apply(t->start, NULL, 0);
    finish_task(t, res);
    abort();
}

DLLEXPORT void julia_init(JL_IMAGE_SEARCH rel)
{
    _julia_init(rel);
}

static void ctx_switch(jl_task_t *t)
{
    if (t == jl_current_task)
        return;
    jl_task_t *lastt = jl_current_task;
    bt_size = 0;  // backtraces don't survive task switches, see e.g. issue #12485
    /*
      making task switching interrupt-safe is going to be challenging.
      we need JL_SIGATOMIC_BEGIN in jl_enter_handler, and then
      JL_SIGATOMIC_END after every JL_TRY sigsetjmp that returns zero.
      also protect jl_eh_restore_state.
      then we need JL_SIGATOMIC_BEGIN at the top of this function (ctx_switch).
      the JL_SIGATOMIC_END at the end of this function handles the case
      of task switching with yieldto().
      then we need to handle the case of task switching via raise().
      to do that, the top of every catch block must do JL_SIGATOMIC_END
      *IF AND ONLY IF* throwing the exception involved a task switch.
    */
    //JL_SIGATOMIC_BEGIN();

    // set up global state for new task
    jl_current_task->gcstack = jl_pgcstack;
    jl_pgcstack = t->gcstack;

    // restore task's current module, looking at parent tasks
    // if it hasn't set one.
    jl_task_t *last = t;
    while (last->current_module == NULL && last != jl_root_task) {
        last = last->parent;
    }
    if (last->current_module != NULL) {
        jl_current_module = last->current_module;
    }

    jl_current_task = t;

    if (jl_setjmp(lastt->ctx, 0)) return; // store the old context, return when execution resumes here

#ifdef COPY_STACKS
    if (lastt->copy_stack) // may need to save the old copy-stack
        save_stack(lastt);
#endif

#ifdef _OS_WINDOWS_
    if (t->fiber != lastt->fiber) {
        SwitchToFiber(t->fiber);
#ifdef COPY_STACKS
        if (!jl_current_task->copy_stack || lastt == jl_current_task)
            return;
#endif
        t = jl_current_task;
    }
#endif

#ifdef COPY_STACKS
    if (t->copy_stack && t->stkbuf) // task already exists, resume from the copy-stack
        restore_stack(t, lastt->copy_stack ? NULL : (void*)(intptr_t)1); // resume at jl_setjmp of the other task after restoring the stack (doesn't return)
#endif

    if (!t->copy_stack && !t->stkbuf) // task not started yet, jump to start_task
        jl_new_fiber(t, t->ssize); // (doesn't return)

    jl_longjmp(t->ctx, 1); // resume at jl_setjmp of the other task (doesn't return)
//JL_SIGATOMIC_END();
}

JL_THREAD jl_value_t * volatile jl_task_arg_in_transit;
extern int jl_in_gc;
DLLEXPORT jl_value_t *jl_switchto(jl_task_t *t, jl_value_t *arg)
{
    if (t == jl_current_task) {
        throw_if_exception_set(t);
        return arg;
    }
    if (t->state == done_sym || t->state == failed_sym ||
        (t->stkbuf == (void*)(intptr_t)-1)) {
        if (t->exception != jl_nothing)
            jl_throw(t->exception);
        return t->result;
    }
    if (jl_in_gc)
        jl_error("task switch not allowed from inside gc finalizer");
    jl_task_arg_in_transit = arg;
    ctx_switch(t);
    jl_value_t *val = jl_task_arg_in_transit;
    jl_task_arg_in_transit = jl_nothing;
    throw_if_exception_set(jl_current_task);
    return val;
}

ptrint_t bt_data[MAX_BT_SIZE+1];
size_t bt_size = 0;

// Always Set *func_name and *file_name to malloc'd pointers (non-NULL)
static int frame_info_from_ip(char **func_name,
                              char **file_name, size_t *line_num,
                              char **inlinedat_file, size_t *inlinedat_line,
                              size_t ip, int skipC, int skipInline)
{
    static const char *name_unknown = "???";
    int fromC = 0;

    jl_getFunctionInfo(func_name, file_name, line_num, inlinedat_file, inlinedat_line, ip, &fromC,
                       skipC, skipInline);
    if (!*func_name) {
        *func_name = strdup(name_unknown);
        *line_num = ip;
    }
    if (!*file_name) {
        *file_name = strdup(name_unknown);
    }
    return fromC;
}

#if defined(_OS_WINDOWS_)
#ifdef _CPU_X86_64_
static UNWIND_HISTORY_TABLE HistoryTable;
#else
static struct {
    DWORD64 dwAddr;
    DWORD64 ImageBase;
} HistoryTable;
#endif
static PVOID CALLBACK JuliaFunctionTableAccess64(
        _In_  HANDLE hProcess,
        _In_  DWORD64 AddrBase)
{
    //jl_printf(JL_STDOUT, "lookup %d\n", AddrBase);
#ifdef _CPU_X86_64_
    DWORD64 ImageBase;
    PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(AddrBase, &ImageBase, &HistoryTable);
    if (fn) return fn;
    if (jl_in_stackwalk) {
        return 0;
    }
    jl_in_stackwalk = 1;
    PVOID ftable = SymFunctionTableAccess64(hProcess, AddrBase);
    jl_in_stackwalk = 0;
    return ftable;
#else
    return SymFunctionTableAccess64(hProcess, AddrBase);
#endif
}
static DWORD64 WINAPI JuliaGetModuleBase64(
        _In_  HANDLE hProcess,
        _In_  DWORD64 dwAddr)
{
    //jl_printf(JL_STDOUT, "lookup base %d\n", dwAddr);
#ifdef _CPU_X86_64_
    DWORD64 ImageBase;
    PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(dwAddr, &ImageBase, &HistoryTable);
    if (fn) return ImageBase;
    if (jl_in_stackwalk) {
        return 0;
    }
    jl_in_stackwalk = 1;
    DWORD64 fbase = SymGetModuleBase64(hProcess, dwAddr);
    jl_in_stackwalk = 0;
    return fbase;
#else
    if (dwAddr == HistoryTable.dwAddr) return HistoryTable.ImageBase;
    DWORD64 ImageBase = jl_getUnwindInfo(dwAddr);
    if (ImageBase) {
        HistoryTable.dwAddr = dwAddr;
        HistoryTable.ImageBase = ImageBase;
        return ImageBase;
    }
    return SymGetModuleBase64(hProcess, dwAddr);
#endif
}

int needsSymRefreshModuleList;
BOOL (WINAPI *hSymRefreshModuleList)(HANDLE);
DLLEXPORT size_t rec_backtrace(ptrint_t *data, size_t maxsize)
{
    CONTEXT Context;
    memset(&Context, 0, sizeof(Context));
    RtlCaptureContext(&Context);
    return rec_backtrace_ctx(data, maxsize, &Context);
}
DLLEXPORT size_t rec_backtrace_ctx(ptrint_t *data, size_t maxsize, CONTEXT *Context)
{
    if (needsSymRefreshModuleList && hSymRefreshModuleList != 0 && !jl_in_stackwalk) {
        jl_in_stackwalk = 1;
        hSymRefreshModuleList(GetCurrentProcess());
        jl_in_stackwalk = 0;
        needsSymRefreshModuleList = 0;
    }
#if !defined(_CPU_X86_64_)
    if (jl_in_stackwalk) {
        return 0;
    }
    DWORD MachineType = IMAGE_FILE_MACHINE_I386;
    STACKFRAME64 stk;
    memset(&stk, 0, sizeof(stk));
    stk.AddrPC.Offset = Context->Eip;
    stk.AddrStack.Offset = Context->Esp;
    stk.AddrFrame.Offset = Context->Ebp;
    stk.AddrPC.Mode = AddrModeFlat;
    stk.AddrStack.Mode = AddrModeFlat;
    stk.AddrFrame.Mode = AddrModeFlat;
    jl_in_stackwalk = 1;
#endif

    size_t n = 0;
    while (n < maxsize) {
#ifndef _CPU_X86_64_
        data[n++] = (intptr_t)stk.AddrPC.Offset;
        BOOL result = StackWalk64(MachineType, GetCurrentProcess(), hMainThread,
            &stk, Context, NULL, JuliaFunctionTableAccess64, JuliaGetModuleBase64, NULL);
        if (!result)
            break;
#else
        data[n++] = (intptr_t)Context->Rip;
        DWORD64 ImageBase = JuliaGetModuleBase64(GetCurrentProcess(), Context->Rip);
        if (!ImageBase)
            break;

        MEMORY_BASIC_INFORMATION mInfo;

        PRUNTIME_FUNCTION FunctionEntry = (PRUNTIME_FUNCTION)JuliaFunctionTableAccess64(GetCurrentProcess(), Context->Rip);
        if (!FunctionEntry) { // assume this is a NO_FPO RBP-based function
            Context->Rsp = Context->Rbp;                 // MOV RSP, RBP

            // Check whether the pointer is valid and executable before dereferencing
            // to avoid segfault while recording. See #10638.
            if (VirtualQuery((LPCVOID)Context->Rsp, &mInfo, sizeof(MEMORY_BASIC_INFORMATION)) == 0)
                break;
            DWORD X = mInfo.AllocationProtect;
            if (!((X&PAGE_READONLY) || (X&PAGE_READWRITE) || (X&PAGE_WRITECOPY) || (X&PAGE_EXECUTE_READ)) ||
                  (X&PAGE_GUARD) || (X&PAGE_NOACCESS))
                break;

            Context->Rbp = *(DWORD64*)Context->Rsp;      // POP RBP
            Context->Rsp = Context->Rsp + sizeof(void*);
            Context->Rip = *(DWORD64*)Context->Rsp;      // POP RIP (aka RET)
            Context->Rsp = Context->Rsp + sizeof(void*);
        }
        else {
            PVOID HandlerData;
            DWORD64 EstablisherFrame;
            (void)RtlVirtualUnwind(
                    0 /*UNW_FLAG_NHANDLER*/,
                    ImageBase,
                    Context->Rip,
                    FunctionEntry,
                    Context,
                    &HandlerData,
                    &EstablisherFrame,
                    NULL);
        }
        if (!Context->Rip)
            break;
#endif
    }
#if !defined(_CPU_X86_64_)
    jl_in_stackwalk = 0;
#endif
    return n;
}
#else
// stacktrace using libunwind
DLLEXPORT size_t rec_backtrace(ptrint_t *data, size_t maxsize)
{
    unw_context_t uc;
    unw_getcontext(&uc);
    return rec_backtrace_ctx(data, maxsize, &uc);
}
DLLEXPORT size_t rec_backtrace_ctx(ptrint_t *data, size_t maxsize, unw_context_t *uc)
{
#if !defined(_CPU_ARM_) && !defined(_CPU_PPC64_)
    unw_cursor_t cursor;
    unw_word_t ip;
    size_t n=0;

    unw_init_local(&cursor, uc);
    do {
        if (n >= maxsize)
            break;
        if (unw_get_reg(&cursor, UNW_REG_IP, &ip) < 0)
            break;
        data[n++] = ip;
    } while (unw_step(&cursor) > 0);
    return n;
#else
    return 0;
#endif
}
#ifdef LIBOSXUNWIND
size_t rec_backtrace_ctx_dwarf(ptrint_t *data, size_t maxsize, unw_context_t *uc)
{
    unw_cursor_t cursor;
    unw_word_t ip;
    size_t n=0;

    unw_init_local_dwarf(&cursor, uc);
    do {
        if (n >= maxsize)
            break;
        if (unw_get_reg(&cursor, UNW_REG_IP, &ip) < 0)
            break;
        data[n++] = ip;
    } while (unw_step(&cursor) > 0);
    return n;
}
#endif
#endif

static void record_backtrace(void)
{
    bt_size = rec_backtrace(bt_data, MAX_BT_SIZE);
}

static jl_value_t *array_ptr_void_type = NULL;
DLLEXPORT jl_value_t *jl_backtrace_from_here(void)
{
    jl_svec_t *tp = NULL;
    jl_array_t *bt = NULL;
    JL_GC_PUSH2(&tp, &bt);
    if (array_ptr_void_type == NULL) {
        tp = jl_svec2(jl_voidpointer_type, jl_box_long(1));
        array_ptr_void_type = jl_apply_type((jl_value_t*)jl_array_type, tp);
    }
    bt = jl_alloc_array_1d(array_ptr_void_type, MAX_BT_SIZE);
    size_t n = rec_backtrace((ptrint_t*)jl_array_data(bt), MAX_BT_SIZE);
    if (n < MAX_BT_SIZE)
        jl_array_del_end(bt, MAX_BT_SIZE-n);
    JL_GC_POP();
    return (jl_value_t*)bt;
}

DLLEXPORT jl_value_t *jl_lookup_code_address(void *ip, int skipC)
{
    char *func_name;
    size_t line_num;
    char *file_name;
    size_t inlinedat_line;
    char *inlinedat_file;
    int fromC = frame_info_from_ip(&func_name, &file_name, &line_num,
                                   &inlinedat_file, &inlinedat_line, (size_t)ip, skipC, 0);
    jl_value_t *r = (jl_value_t*)jl_alloc_svec(7);
    JL_GC_PUSH1(&r);
    jl_svecset(r, 0, jl_symbol(func_name));
    jl_svecset(r, 1, jl_symbol(file_name));
    jl_svecset(r, 2, jl_box_long(line_num));
    jl_svecset(r, 3, jl_symbol(inlinedat_file ? inlinedat_file : ""));
    jl_svecset(r, 4, jl_box_long(inlinedat_file ? inlinedat_line : -1));
    jl_svecset(r, 5, jl_box_bool(fromC));
    jl_svecset(r, 6, jl_box_long((intptr_t)ip));
    free(func_name);
    free(file_name);
    free(inlinedat_file);
    JL_GC_POP();
    return r;
}

DLLEXPORT jl_value_t *jl_get_backtrace(void)
{
    jl_svec_t *tp = NULL;
    jl_array_t *bt = NULL;
    JL_GC_PUSH2(&tp, &bt);
    if (array_ptr_void_type == NULL) {
        tp = jl_svec2(jl_voidpointer_type, jl_box_long(1));
        array_ptr_void_type = jl_apply_type((jl_value_t*)jl_array_type, tp);
    }
    bt = jl_alloc_array_1d(array_ptr_void_type, bt_size);
    memcpy(bt->data, bt_data, bt_size*sizeof(void*));
    JL_GC_POP();
    return (jl_value_t*)bt;
}

//for looking up functions from gdb:
DLLEXPORT void gdblookup(ptrint_t ip)
{
    char *func_name;
    size_t line_num;
    char *file_name;
    size_t inlinedat_line;
    char *inlinedat_file;
    frame_info_from_ip(&func_name, &file_name, &line_num, &inlinedat_file, &inlinedat_line, ip,
                      /* skipC */ 0, /* skipInline */ 1);
    if (line_num == ip) {
        jl_safe_printf("unknown function (ip: %p)\n", (void*)ip);
    }
    else if (line_num == -1) {
        jl_safe_printf("%s at %s (unknown line)\n", func_name, file_name);
    }
    else {
        jl_safe_printf("%s at %s:%" PRIuPTR "\n", func_name, file_name,
                       (uintptr_t)line_num);
    }
    free(func_name);
    free(file_name);
    free(inlinedat_file);
}

DLLEXPORT void jlbacktrace()
{
    size_t n = bt_size; //bt_size > 40 ? 40 : bt_size;
    for(size_t i=0; i < n; i++)
        gdblookup(bt_data[i]);
}

DLLEXPORT void gdbbacktrace()
{
    record_backtrace();
    jlbacktrace();
}


// yield to exception handler
void NORETURN throw_internal(jl_value_t *e)
{
    assert(e != NULL);
    jl_exception_in_transit = e;
    if (jl_current_task->eh != NULL) {
        jl_longjmp(jl_current_task->eh->eh_ctx, 1);
    }
    else {
        if (jl_current_task == jl_root_task) {
            jl_printf(JL_STDERR, "fatal: error thrown and no exception handler available.\n");
            jl_static_show(JL_STDERR, e);
            jl_printf(JL_STDERR, "\n");
            jlbacktrace();
            jl_exit(1);
        }
        jl_current_task->exception = e;
        jl_gc_wb(jl_current_task, e);
        finish_task(jl_current_task, e);
        assert(0);
    }
    assert(0);
}

// record backtrace and raise an error
DLLEXPORT void jl_throw(jl_value_t *e)
{
    assert(e != NULL);
    record_backtrace();
    throw_internal(e);
}

DLLEXPORT void jl_rethrow(void)
{
    throw_internal(jl_exception_in_transit);
}

DLLEXPORT void jl_rethrow_other(jl_value_t *e)
{
    throw_internal(e);
}

DLLEXPORT void jl_throw_with_superfluous_argument(jl_value_t *e, int line)
{
    jl_throw(e);
}

jl_value_t *jl_task_cleanup_func;

DLLEXPORT jl_task_t *jl_new_task(jl_function_t *start, size_t ssize)
{
    size_t pagesz = jl_page_size;
    jl_task_t *t = (jl_task_t*)jl_gc_allocobj(sizeof(jl_task_t));
    jl_set_typeof(t, jl_task_type);
#ifndef COPY_STACKS
    if (ssize == 0) // unspecified -- pick some default size
        ssize = JL_STACK_SIZE;
#endif
    if (ssize != 0) {
        if (ssize < MINSIGSTKSZ)
            ssize = MINSIGSTKSZ;
        ssize = LLT_ALIGN(ssize, pagesz);
    }
    t->ssize = ssize;
    t->current_module = NULL;
    t->parent = jl_current_task;
    t->tls = jl_nothing;
    t->consumers = jl_nothing;
    t->state = runnable_sym;
    t->start = start;
    t->result = jl_nothing;
    t->donenotify = jl_nothing;
    t->exception = jl_nothing;
    t->backtrace = jl_nothing;
    // there is no active exception handler available on this stack yet
    t->eh = NULL;
    t->gcstack = NULL;
    t->stkbuf = NULL;
    t->copy_stack = (ssize == 0);
#if defined(JL_DEBUG_BUILD)
    if (!t->copy_stack)
        memset(&t->ctx, 0, sizeof(t->ctx));
#endif
#ifdef COPY_STACKS
    if (t->copy_stack)
        memcpy(&t->ctx, &jl_basectx, sizeof(t->ctx));
#endif

#ifdef _OS_WINDOWS_
    t->fiber = NULL;
#ifdef COPY_STACKS
    if (t->copy_stack)
        t->fiber = jl_basefiber;
#endif
#endif

    jl_gc_add_finalizer((jl_value_t*)t, (jl_function_t*)jl_task_cleanup_func);
    return t;
}

static void jl_task_cleanup(jl_task_t *t)
{
    void *stk = t->stkbuf;
    if (stk && stk != (void*)(intptr_t)-1) {
        t->stkbuf = (void*)(intptr_t)-1;
        if (t->copy_stack)
            free(stk - sizeof(size_t));
        else
#ifdef _OS_WINDOWS_
            DeleteFiber(t->fiber);
#else
            free_stack(stk, t->ssize);
#endif
    }
}

DLLEXPORT jl_value_t *jl_get_current_task(void)
{
    return (jl_value_t*)jl_current_task;
}

// Do one-time initializations for task system
void jl_init_tasks(void)
{
    _probe_arch();
    jl_task_type = jl_new_datatype(jl_symbol("Task"),
                                   jl_any_type,
                                   jl_emptysvec,
                                   jl_svec(9,
                                            jl_symbol("parent"),
                                            jl_symbol("storage"),
                                            jl_symbol("state"),
                                            jl_symbol("consumers"),
                                            jl_symbol("donenotify"),
                                            jl_symbol("result"),
                                            jl_symbol("exception"),
                                            jl_symbol("backtrace"),
                                            jl_symbol("code")),
                                   jl_svec(9,
                                            jl_any_type,
                                            jl_any_type, jl_sym_type,
                                            jl_any_type, jl_any_type,
                                            jl_any_type, jl_any_type,
                                            jl_any_type, jl_function_type),
                                   0, 1, 8);
    jl_svecset(jl_task_type->types, 0, (jl_value_t*)jl_task_type);

    done_sym = jl_symbol("done");
    failed_sym = jl_symbol("failed");
    runnable_sym = jl_symbol("runnable");

    jl_task_cleanup_func = jl_box_voidpointer(&jl_task_cleanup);
}


#ifdef _OS_WINDOWS_
#ifdef COPY_STACKS
static VOID NOINLINE NORETURN CALLBACK start_basefiber(PVOID lpParameter)
{
    jl_stackbase = &lpParameter;
    if (jl_setjmp(jl_basectx, 0))
        start_task();
    SwitchToFiber(jl_current_task->fiber);
    start_task();
}
#endif
static VOID NOINLINE NORETURN CALLBACK start_fiber(PVOID lpParameter)
{
    jl_current_task->stkbuf = &lpParameter;
    start_task();
}
static void jl_new_fiber(jl_task_t *t, size_t ssize)
{
    LPVOID jl_fiber = CreateFiberEx(ssize, ssize, FIBER_FLAG_FLOAT_SWITCH, start_fiber, NULL);
    if (jl_fiber == NULL)
        jl_error("CreateFiberEx failed");
    t->fiber = jl_fiber;
}
static void jl_init_basefiber(size_t ssize)
{
    jl_current_task->fiber = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (jl_current_task->fiber == NULL)
        jl_error("ConvertThreadToFiberEx failed");
#ifdef COPY_STACKS
    jl_basefiber = CreateFiberEx(ssize, ssize, FIBER_FLAG_FLOAT_SWITCH, start_basefiber, NULL);
    if (jl_basefiber == NULL)
        jl_error("CreateFiberEx failed");
    SwitchToFiber(jl_basefiber); /* initializes jl_stackbase and jl_basectx */
#endif
}
#endif

#if defined(HAVE_UCONTEXT)
static void start_basefiber()
{
#ifdef COPY_STACKS
    if (jl_setjmp(jl_basectx, 0))
        start_task();
#else
    abort(); // function not used
#endif
}
static void jl_new_fiber(jl_task_t *t, size_t ssize)
{
    char *stk = malloc_stack(ssize);
    jl_base_uctx.uc_stack.ss_sp = stk;
    jl_base_uctx.uc_stack.ss_size = ssize;
    if (t != jl_root_task) {
        jl_base_uctx.uc_link = NULL;
        t->stkbuf = stk;
        makecontext(&jl_base_uctx, &start_task, 0);
        setcontext(&jl_base_uctx);
        abort();
    }
    else {
        ucontext_t jl_root_uctx;
        jl_stackbase = stk + JL_STACK_SIZE;
        jl_base_uctx.uc_link = &jl_root_uctx;
        makecontext(&jl_base_uctx, &start_basefiber, 0);
        swapcontext(&jl_root_uctx, &jl_base_uctx); // initializes jl_basectx
    }
}
static void jl_init_basefiber(size_t ssize)
{
    int r = getcontext(&jl_base_uctx);
    if (r != 0)
        jl_error("getcontext failed");
#ifdef COPY_STACKS
    jl_new_fiber(jl_root_task, ssize);
#endif
}
#endif

#if defined(HAVE_UNW_CONTEXT)
static void start_basefiber()
{
#ifdef COPY_STACKS
    if (jl_setjmp(jl_basectx, 0))
        start_task();
    jl_longjmp(jl_root_task->ctx, 1);
#else
    abort(); // function not used
#endif
}
#if defined(_CPU_X86_) || defined(_CPU_X86_64_)
#define PUSH_RET(ctx, stk) \
    do { \
        stk -= sizeof(uintptr_t); \
        *(uintptr_t*)stk = 0; /* push null RIP/EIP onto the stack */ \
    } while (0)
#elif defined(_CPU_ARM_)
#define PUSH_RET(ctx, stk) \
    unw_set_reg(ctx, UNW_ARM_R14, 0) /* put NULL into the LR */
#else
#error please define how to simulate a CALL on this platform
#endif
static void jl_new_fiber(jl_task_t *t, size_t ssize)
{
    char *stkbuf = malloc_stack(ssize);
    char *stk = stkbuf;
    stk += ssize;
    PUSH_RET(&jl_basecursor, stk);
    if (unw_set_reg(&jl_basecursor, UNW_REG_SP, (uintptr_t)stk) != 0)
        jl_error("unw_set_reg UNW_REG_SP failed");
    uintptr_t fn;
    if (t == jl_root_task)
        fn = (uintptr_t)&start_basefiber;
    else
        fn = (uintptr_t)&start_task;
    if (unw_set_reg(&jl_basecursor, UNW_REG_IP, fn) != 0)
        jl_error("unw_set_reg UNW_REG_IP failed");
    if (t == jl_root_task)
        jl_stackbase = stkbuf + JL_STACK_SIZE;
    else
        t->stkbuf = stkbuf;
    unw_resume(&jl_basecursor); // (doesn't return)
    abort();
}
static void jl_init_basefiber(size_t ssize)
{
    int r = unw_getcontext(&jl_base_uctx);
    if (r != 0)
        jl_error("unw_getcontext failed");
    r = unw_init_local(&jl_basecursor, &jl_base_uctx);
    if (r != 0)
        jl_error("unw_init_local failed");
#ifdef COPY_STACKS
    if (jl_setjmp(jl_root_task->ctx, 0)) return;
    jl_new_fiber(jl_root_task, ssize); // (doesn't return)
#endif
}
#endif

#if defined(HAVE_ASM)
static void start_basefiber()
{
#ifdef COPY_STACKS
    if (jl_setjmp(jl_basectx, 0))
        start_task();
    jl_longjmp(jl_root_task->ctx, 1);
#else
    abort(); // function not used
#endif
}
static void jl_new_fiber(jl_task_t *t, size_t ssize)
{
    char *stkbuf = malloc_stack(ssize);
    char *stk = stkbuf;
    stk += ssize;
    uintptr_t fn;
    if (t == jl_root_task) {
        fn = (uintptr_t)&start_basefiber;
        jl_stackbase = stkbuf + JL_STACK_SIZE;
    }
    else  {
        fn = (uintptr_t)&start_task;
        t->stkbuf = stkbuf;
    }
#ifdef _CPU_X86_64_
#ifdef _OS_WINDOWS_
    stk -= 0x20;
#endif
    asm volatile (
        " movq %0, %%rsp;\n"
        " movq %1, %%rax;\n"
        " xorq %%rbp, %%rbp;\n"
        " push %%rbp;\n" // instead of RSP
        " jmpq *%%rax;\n" // call stack_task with fake stack frame
        " ud2"
        : : "r"(stk), "r"(fn) : "memory" );
#elif defined(_CPU_X86_)
    asm volatile (
        " movl %0, %%esp;\n"
        " movl %1, %%eax;\n"
        " xorl %%ebp, %%ebp;\n"
        " push %%ebp;\n" // instead of ESP
        " jmpl *%%eax;\n" // call stack_task with fake stack frame
        " ud2"
        : : "r"(stk), "r"(fn) : "memory" );
#else
#error HAVE_ASM not implemented for this CPU type
#endif
    __builtin_unreachable();
}
static void jl_init_basefiber(size_t ssize)
{
#ifdef COPY_STACKS
    if (jl_setjmp(jl_root_task->ctx, 0)) return;
    jl_new_fiber(jl_root_task, ssize); // (doesn't return)
#endif
}
#endif

#if defined(HAVE_SIGALTSTACK)
static void start_fiber()
{
    if (jl_setjmp(jl_current_task->ctx, 0))
        start_task();
}
static void jl_new_fiber(jl_task_t *t, size_t ssize)
{
    stack_t uc_stack, osigstk;
    struct sigaction sa, osa;
    sigset_t set, oset;
    char *stk = malloc_stack(ssize);
    // setup
    sigfillset(&set);
    if (sigprocmask(SIG_BLOCK, &set, &oset) != 0)
       jl_error("sigprocmask failed");
    uc_stack.ss_sp = stk;
    uc_stack.ss_size = ssize;
    uc_stack.ss_flags = 0;
    if (sigaltstack(&uc_stack, &osigstk) != 0)
       jl_error("sigaltstack failed");
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = start_fiber;
    sa.sa_flags = SA_ONSTACK;
    if (sigaction(SIGUSR2, &sa, &osa) != 0)
       jl_error("sigaction failed");
    // emit signal
    pthread_kill(pthread_self(), SIGUSR2); // initializes jl_basectx
    sigdelset(&set, SIGUSR2);
    sigsuspend(&set);
    // cleanup
    if (sigaction(SIGUSR2, &osa, NULL) != 0)
       jl_error("sigaction failed");
    if (osigstk.ss_size < MINSIGSTKSZ && (osigstk.ss_flags | SS_DISABLE))
       osigstk.ss_size = MINSIGSTKSZ;
    if (sigaltstack(&osigstk, NULL) != 0)
       jl_error("sigaltstack failed");
    if (sigprocmask(SIG_SETMASK, &oset, NULL) != 0)
       jl_error("sigprocmask failed");
    if (t == jl_root_task)
        jl_stackbase = stk + JL_STACK_SIZE;
    else
        t->stkbuf = stk;
    if (t != jl_root_task)
        jl_longjmp(t->ctx, 1);
}
static void jl_init_basefiber(size_t ssize)
{
#ifdef COPY_STACKS
    jl_new_fiber(jl_root_task, ssize);
    memcpy(jl_basectx, jl_current_task->ctx, sizeof(jl_basectx));
#endif
}
#endif

// Initialize a root task using the given stack.
void jl_init_root_task(void *stack, size_t ssize)
{
    jl_current_task = (jl_task_t*)jl_gc_allocobj(sizeof(jl_task_t));
    jl_set_typeof(jl_current_task, jl_task_type);

    jl_current_task->copy_stack = 0;
    jl_current_task->stkbuf = stack - ssize - 3000000; // guess at address of bottom of stack
    jl_current_task->ssize = ssize + 3000000; // guess at sizeof stack
    jl_current_task->parent = jl_current_task;
    jl_current_task->current_module = jl_current_module;
    jl_current_task->tls = jl_nothing;
    jl_current_task->consumers = jl_nothing;
    jl_current_task->state = runnable_sym;
    jl_current_task->start = NULL;
    jl_current_task->result = jl_nothing;
    jl_current_task->donenotify = jl_nothing;
    jl_current_task->exception = jl_nothing;
    jl_current_task->backtrace = jl_nothing;
    jl_current_task->eh = NULL;
    jl_current_task->gcstack = NULL;

    jl_root_task = jl_current_task;

    jl_exception_in_transit = (jl_value_t*)jl_nothing;
    jl_task_arg_in_transit = (jl_value_t*)jl_nothing;

    jl_init_basefiber(JL_STACK_SIZE);
}

#ifdef __cplusplus
}
#endif
