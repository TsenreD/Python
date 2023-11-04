import builtins
import dis
import types
import typing as tp

tmp = False


def debug(*args: tp.Any) -> None:
    global tmp
    nargs = list(map(str, args))
    file = open('debug.txt', 'a' if tmp else 'w')
    file.write(" ".join(nargs))
    file.close()
    tmp = True


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value: tp.List[tp.Any] | None = None
        self.bytecode_counter = 0
        self.offsets = \
            {instr.offset: instr for instr in dis.get_instructions(frame_code)}
        debug(self.offsets.keys())

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def topn(self, n: int) -> tp.Any:
        """
        Return a list of n elements from the stack, deepest first.
        """
        if n > 0:
            return self.data_stack[-n:]
        else:
            return []

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        max_offset = max(self.offsets.keys())
        while self.bytecode_counter <= max_offset:
            cur = self.offsets[self.bytecode_counter]
            debug(cur.opname + " " + str(self.bytecode_counter) + " ")
            res = getattr(self, cur.opname.lower() + "_op")(cur.argval)
            debug("finished" + "\n")
            if res:
                if self.return_value:
                    self.return_value.append(res)
                else:
                    self.return_value = [res]
            self.__increase_counter(cur.opname)

        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        arguments = self.popn(arg)
        self_or_callable = self.pop()
        null_or_callable = self.pop() if self.data_stack else None
        if null_or_callable:
            self.push(null_or_callable(self_or_callable, *arguments))
        else:
            self.push(self_or_callable(*arguments))

    def unpack_sequence_op(self, count: int) -> None:
        vals = list(self.pop())
        assert len(vals) == count
        for elem in reversed(vals):
            self.push(elem)

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        for scope in [self.locals, self.globals, self.builtins]:
            if arg in scope:
                self.push(scope[arg])
                return
        raise NameError

    def nop_op(self, arg: str) -> None:
        pass

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        for scope in [self.globals, self.builtins]:
            if arg in scope:
                self.push(scope[arg])
                return
        raise NameError

    def delete_global_op(self, arg: str) -> None:
        if arg in self.globals.keys():
            del self.globals[arg]
        elif arg in self.builtins.keys():
            del self.builtins[arg]
        else:
            raise NameError(f'key {arg} is undefined')

    def __binary_helper(self, arg: str, first: tp.Any, second: tp.Any) -> tp.Any:
        match int(arg):
            case 0:
                return first + second
            case 10:
                return first - second
            case 5:
                return first * second
            case 11:
                return first / second
            case 2:
                return first // second
            case 6:
                return first % second
            case 4:
                return first @ second
            case 8:
                return first ** second

            case 3:
                return first << second
            case 9:
                return first >> second
            case 1:
                return first & second
            case 7:
                return first | second
            case 12:
                return first ^ second

            case 13:
                first += second
                return first
            case 23:
                first -= second
                return first
            case 18:
                first *= second
                return first
            case 24:
                first /= second
                return first
            case 15:
                first //= second
                return first
            case 19:
                first %= second
                return first
            case 17:
                first @= second
                return first
            case 21:
                first **= second
                return first
            case 16:
                first <<= second
                return first
            case 22:
                first >>= second
                return first
            case 14:
                first &= second
                return first
            case 20:
                first |= second
                return first
            case 25:
                first ^= second
                return first
        raise ValueError

    def binary_op_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        second = self.pop()
        first = self.pop()
        res = self.__binary_helper(arg, first, second)
        self.push(res)

    def compare_op_op(self, op: str) -> None:
        left, right = self.popn(2)
        debug("Compare op:", op, "args:", left, right, "\n")
        match op:
            case "==":
                res = left == right
            case "!=":
                res = left != right
            case "in":
                res = left in right
            case "not in":
                res = left not in right
            case ">":
                res = left > right
            case ">=":
                res = left >= right
            case "<":
                res = left < right
            case "<=":
                res = left <= right
            case _:
                raise ValueError
        self.push(res)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        if self.return_value is None:
            self.return_value = self.pop()

    def return_generator_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.push(None)

    def yield_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        return self.top()

    def get_yield_from_iter_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        if self.top() is not None:
            self.push(self.pop().__iter__())

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def make_function_op(self, flag: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code: types.CodeType = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args = {code.co_varnames[i]: args[i]
                           for i in range(code.co_argcount)}

            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_GLOBAL
        """
        const = self.pop()
        self.globals[arg] = const

    def get_iter_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-GET_ITER
        """
        last = self.pop()
        self.push(iter(last))

    def contains_op_op(self, inv: bool) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-GET_ITER
        """
        left, right = self.popn(2)
        self.push(left in right if not inv else left not in right)

    def is_op_op(self, inv: bool) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-GET_ITER
        """
        left, right = self.popn(2)
        self.push(left is right if not inv else left is not right)

    def for_iter_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-FOR_ITER
        """
        it = self.top()
        try:
            debug(str(it))
            self.push(it.__next__())
            debug(" -success \n")
        except StopIteration:
            self.pop()
            self.__jump(arg - 2)

    def pop_jump_forward_if_false_op(self, arg: int) -> None:
        if not self.pop():
            self.__jump(arg)
        else:
            self.bytecode_counter += 2

    def jump_if_false_or_pop_op(self, target: int) -> None:
        top = self.top()
        if not top:
            self.__jump(target)
        else:
            self.pop()
            self.bytecode_counter += 2

    def jump_if_true_or_pop_op(self, target: int) -> None:
        top = self.top()
        debug("Jumping if true", target, top)
        if top:
            self.__jump(target)
        else:
            self.pop()
            self.bytecode_counter += 2

    def load_build_class_op(self, ignored: tp.Any) -> None:
        self.push(__build_class__)

    def pop_jump_forward_if_none_op(self, arg: int) -> None:
        if self.pop() is None:
            self.__jump(arg)
        else:
            self.bytecode_counter += 2

    def jump_forward_op(self, actual: int) -> None:
        if actual <= self.bytecode_counter:
            raise ValueError("Jumping backward in forward jump")
        self.__jump(actual)

    def pop_jump_forward_if_true_op(self, arg: int) -> None:
        debug("Forward jump:", arg, self.top(), "\n")
        if self.pop():
            self.__jump(arg)
        else:
            self.bytecode_counter += 2

    def jump_backward_op(self, actual: int) -> None:
        if actual >= self.bytecode_counter:
            raise ValueError("Jumping forward in backward jump")
        self.__jump(actual)

    def __jump(self, actual: int) -> None:
        self.bytecode_counter = actual

    def build_slice_op(self, argc: int) -> None:
        if argc == 2:
            tos1, tos = self.popn(2)
            self.push(slice(tos1, tos))
        elif argc == 3:
            tos2, tos1, tos = self.popn(3)
            self.push(slice(tos2, tos1, tos))
        else:
            raise ValueError(f"argc must be 2 or 3, got {argc}")

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def build_list_op(self, count: int) -> None:
        self.push(list(self.popn(count)))

    def list_extend_op(self, idx: int) -> None:
        tos = self.pop()
        list.extend(self.data_stack[-idx], tos)

    def list_append_op(self, idx: int) -> None:
        tos = self.pop()
        list.append(self.data_stack[-idx], tos)

    def map_add_op(self, idx: int) -> None:
        s_0, value = self.popn(2)
        self.data_stack[-idx].__setitem__(s_0, value)

    def set_add_op(self, idx: int) -> None:
        value = self.pop()
        self.data_stack[-idx].add(value)

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def set_update_op(self, idx: int) -> None:
        tos = self.pop()
        set.update(self.data_stack[-idx], tos)

    def build_map_op(self, count: int) -> None:
        entries = self.popn(2 * count)
        keys = entries[::2]
        values = entries[1::2]
        res = {key: value for key, value in zip(keys, values)}
        self.push(res)

    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        values = self.popn(count)
        res = {key: value for key, value in zip(keys, values)}
        self.push(res)

    def format_value_op(self, flag: tuple[tp.Any, tp.Any]) -> None:
        match flag[0]:
            case builtins.str:
                self.data_stack[-1] = str(self.data_stack[-1])
            case builtins.repr:
                self.data_stack[-1] = repr(self.data_stack[-1])
            case builtins.ascii:
                self.data_stack[-1] = ascii(self.data_stack[-1])
            case None:
                return
            case _:
                raise ValueError("Check formatting")

    def build_string_op(self, count: int) -> None:
        strings = self.popn(count)
        self.push("".join(strings))

    def load_assertion_error_op(self, arg: str) -> None:
        self.push(AssertionError(arg))

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            raise
        elif argc == 1:
            raise self.pop()
        elif argc == 2:
            tos1, tos = self.popn(2)
            raise tos1 from tos

    def store_subscr_op(self, arg: tp.Any) -> None:
        tos2, tos1, tos = self.topn(3)
        tos1[tos] = tos2

    def delete_subscr_op(self, arg: tp.Any) -> None:
        tos1, tos = self.popn(2)
        del tos1[tos]

    def binary_subscr_op(self, arg: tp.Any) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1[tos])

    def unary_negative_op(self, arg: tp.Any) -> None:
        self.data_stack[-1] = -self.top()

    def unary_positive_op(self, arg: tp.Any) -> None:
        self.data_stack[-1] = +self.top()

    def unary_not_op(self, arg: tp.Any) -> None:
        self.data_stack[-1] = not self.top()

    def unary_invert_op(self, arg: tp.Any) -> None:
        self.data_stack[-1] = ~self.top()

    def store_attr_op(self, name: str) -> None:
        top1, top = self.popn(2)
        setattr(top, name, top1)
        self.push(top)

    def store_fast_op(self, name: str) -> None:
        debug(name + " stored")
        self.locals[name] = self.pop()

    def load_fast_op(self, name: str) -> None:
        if name in self.locals:
            self.push(self.locals[name])
        else:
            raise UnboundLocalError(f'encountered unknown variable {name}')

    def load_method_op(self, name: str) -> None:
        tos = self.pop()
        if hasattr(tos, name):
            func = getattr(tos, name)
            self.push(None)
            self.push(func)
        else:
            self.push(None)

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def load_attr_op(self, name: str) -> None:
        self.data_stack[-1] = getattr(self.top(), name)

    def delete_attr_op(self, name: str) -> None:
        if hasattr(self.top(), name):
            delattr(self.top(), name)

    def delete_name_op(self, name: str) -> None:
        for scope in [self.locals, self.globals, self.builtins]:
            if name in scope:
                del scope[name]
                return
        raise NameError(f"Name {name} not found in any scope")

    def copy_op(self, idx: int) -> None:
        self.push(self.data_stack[-idx])

    def swap_op(self, idx: int) -> None:
        st = self.data_stack
        st[-idx], st[-1] = st[-1], st[-idx]

    def import_name_op(self, name: str) -> None:
        tos1, tos = self.popn(2)
        self.push(__import__(name, self.globals, self.locals, tos, tos1))

    def import_from_op(self, name: str) -> None:
        module = self.top()
        self.push(getattr(module, name))

    def import_star_op(self, ignored: str) -> None:
        module = self.top()
        for symb in dir(module):
            if not symb.startswith('__'):
                self.locals[symb] = getattr(module, symb)

    def __increase_counter(self, name: str) -> None:
        if "JUMP" in name:
            return
        if name in ["CALL", "BINARY_SUBSCR", "STORE_ATTR", "LOAD_ATTR"]:
            self.bytecode_counter += 10
        elif name == "COMPARE_OP":
            self.bytecode_counter += 6
        elif name == "LOAD_GLOBAL":
            self.bytecode_counter += 12
        elif name == "LOAD_METHOD":
            self.bytecode_counter += 22
        elif (name in
              ["PRECALL", "BINARY_OP", "UNPACK_SEQUENCE", "STORE_SUBSCR"]):
            self.bytecode_counter += 4
        else:
            self.bytecode_counter += 2


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'],
                      globals_context, globals_context)
        return frame.run()
