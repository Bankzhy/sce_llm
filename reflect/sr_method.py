import copy
import json

# Node
VAR_IDENTIFIER = "var"
VAR_ASSIGNMENT = "var_assignment"
METHOD_IDENTIFIER = "method"
DATATYPE = "datatype"
CONCEPT = "concept"

# Rels
TYPE_OF = "type_of"
RELATED_CONCEPT = "related_concept"
CONTROL_DEPENDENCY = "control_dependency"
DATA_DEPENDENCY = "data_dependency"
HAS_METHOD = "has_method"
HAS_PROPERTY = "has_property"
ASSIGNMENT = "assignment"

from reflect.sr_core import SRCore
from reflect.sr_statement import SRIFStatement, SRFORStatement, SRWhileStatement, SRTRYStatement, SRStatement, \
    SRSwitchStatement


class SRMethod(SRCore):
    def __init__(self, id="-1", word_list=[], text="", method_name="", param_list=[], return_type="",
                 statement_list=[], modifiers=[], throws=[]):
        self.word_list = word_list
        self.method_name = method_name
        self.statement_list = statement_list
        self.param_list = param_list
        self.return_type = return_type
        self.id = id
        self.modifiers = modifiers
        self.throws = throws
        self.comment = ""
        self.text = text
        self.mkg = None

    def get_method_string(self):
        method_text = " ".join(self.word_list)
        method_text += "("
        for index, p in enumerate(self.param_list):
            method_text += p.get_string()
            if index != len(self.param_list)-1:
                method_text += ", "
        method_text += ")"

        method_text += " {\n"
        for st in self.statement_list:
            method_text += "    "
            method_text += st.get_statement_string()
            method_text += "\n"
        method_text += "}"
        return method_text

    def to_string(self, space=1):
        result = ""
        result += " ".join(self.modifiers)
        result += " "
        result += self.return_type
        result += " "
        result += self.method_name

        result += "("
        if len(self.param_list) > 0:
            result += ",".join(
                map(lambda p: p.to_string(), self.param_list)
            )
        result += ")"
        result += " "
        result += " ".join(self.throws)
        result += "{"
        result += "\n"

        space += 1
        if len(self.statement_list) > 0:
            for statement in self.statement_list:
                for x in range(0, space):
                    result += "    "
                result += statement.to_string(space=space)
                result += "\n"
        result += "\n"

        for x in range(0, space-1):
            result += "    "
        result += "}"
        return result
        # return " ".join(self.word_list)

    def get_method_dic(self):
        method_info = {}
        method_info["methodName"] = self.method_name
        method_info["returnType"] = self.return_type
        method_info["sid"] = self.id
        return method_info

    def __get_all_statement(self, statement_list, exclude_special=True):
        result = []
        for statement in statement_list:
            if type(statement) == SRIFStatement:
                if exclude_special is False:
                    result.append(statement)
                result.extend(self.__get_all_statement(statement.pos_statement_list, exclude_special))
                result.extend(self.__get_all_statement(statement.neg_statement_list, exclude_special))
            elif type(statement) == SRFORStatement:
                if exclude_special is False:
                    result.append(statement)
                result.extend(self.__get_all_statement(statement.child_statement_list, exclude_special))
            elif type(statement) == SRWhileStatement:
                if exclude_special is False:
                    result.append(statement)
                result.extend(self.__get_all_statement(statement.child_statement_list, exclude_special))
            elif type(statement) == SRTRYStatement:
                if exclude_special is False:
                    result.append(statement)
                result.extend(self.__get_all_statement(statement.try_statement_list, exclude_special))
                for cb in statement.catch_block_list:
                    result.extend(self.__get_all_statement(cb.child_statement_list, exclude_special))
                result.extend(self.__get_all_statement(statement.final_block_statement_list, exclude_special))
            else:
                result.append(statement)
        return result

    def get_all_statement(self, exclude_special=True):
        return self.__get_all_statement(self.statement_list, exclude_special)

    def get_all_local_var(self):
        all_statement = self.__get_all_statement(self.statement_list, exclude_special=False)
        result = []
        for st in all_statement:
            # print(st.word_list)
            result.extend(st.get_loc_var_list())

        # for st in all_statement:
        #     if type(st) == SRStatement:
        #         if "=" in st.word_list:
        #             if st.word_list[st.word_list.index("=")-1] not in result:
        #                 result.append(st.word_list[st.word_list.index("=")-1])

        return result

    def get_all_fields(self, field_name_list):
        result_list = []
        for statement in self.statement_list:
            st_f_l = statement.get_all_fields(field_name_list)
            result_list.extend(st_f_l)
        return result_list

    def get_all_method_used(self, method_name_list):
        result_list = []
        for statement in self.statement_list:
            st_m_l = statement.get_all_method_used(method_name_list)
            result_list.extend(st_m_l)
        return result_list

    def get_all_word(self):
        all_statement = self.__get_all_statement(self.statement_list, exclude_special=False)
        result = []
        for st in all_statement:
            # print(st.word_list)
            result.extend(st.get_all_word())
        return result

    def get_method_LOC(self):
        return len(self.__get_all_statement(self.statement_list, exclude_special=False))

    def __check_replace_statement(self, statement_list, new_statement_list, statement_id):
        front_st_list = []
        back_st_list = []
        is_front = True
        result_list = front_st_list
        for st in statement_list:
            if st.id == statement_id:
                is_front = False
            else:
                if is_front:
                    front_st_list.append(st)
                else:
                    back_st_list.append(st)
        if len(back_st_list) > 0:
            result_list.extend(front_st_list)
            result_list.extend(new_statement_list)
            result_list.extend(back_st_list)
        return result_list

    def __insert_st_list(self, statement_id, statement_list, new_statement_list):
        result_list = []
        for statement in statement_list:
            if type(statement) == SRIFStatement:
                statement.pos_statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=statement.pos_statement_list, new_statement_list=new_statement_list)
                statement.neg_statement_list= self.__insert_st_list(statement_id=statement_id, statement_list=statement.neg_statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRSwitchStatement:
                for sc in statement.switch_case_list:
                    sc.statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=sc.statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRFORStatement:
                statement.child_statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=statement.child_statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRWhileStatement:
                statement.child_statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=statement.child_statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRTRYStatement:
                statement.try_statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=statement.try_statement_list, new_statement_list=new_statement_list)
                for cb in statement.catch_block_list:
                    cb.child_statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=cb.child_statement_list, new_statement_list=new_statement_list)
                statement.final_block_statement_list = self.__insert_st_list(statement_id=statement_id, statement_list=statement.final_block_statement_list, new_statement_list=new_statement_list)
            # else:
            if str(statement.id) == str(statement_id):
                result_list.extend(new_statement_list)
                result_list.append(statement)
            else:
                result_list.append(statement)
        return result_list

    def __find_replace_st_list(self, statement_id, statement_list, new_statement_list):
        front_st_list = []
        back_st_list = []
        is_front = True
        result_list = []

        for statement in statement_list:
            if type(statement) == SRIFStatement:
                statement.pos_statement_list = self.__find_replace_st_list(statement_id=statement_id, statement_list=statement.pos_statement_list, new_statement_list=new_statement_list)
                statement.neg_statement_list= self.__find_replace_st_list(statement_id=statement_id, statement_list=statement.neg_statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRSwitchStatement:
                for sc in statement.switch_case_list:
                    sc.statement_list = self.__find_replace_st_list(statement_id=statement_id,
                                                              statement_list=sc.statement_list,
                                                              new_statement_list=new_statement_list)

            elif type(statement) == SRFORStatement:
                statement.child_statement_list = self.__find_replace_st_list(statement_id=statement_id, statement_list=statement.child_statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRWhileStatement:
                statement.child_statement_list = self.__find_replace_st_list(statement_id=statement_id, statement_list=statement.child_statement_list, new_statement_list=new_statement_list)
            elif type(statement) == SRTRYStatement:
                statement.try_statement_list = self.__find_replace_st_list(statement_id=statement_id, statement_list=statement.try_statement_list, new_statement_list=new_statement_list)
                for cb in statement.catch_block_list:
                    cb.child_statement_list = self.__find_replace_st_list(statement_id=statement_id, statement_list=cb.child_statement_list, new_statement_list=new_statement_list)
                statement.final_block_statement_list = self.__find_replace_st_list(statement_id=statement_id, statement_list=statement.final_block_statement_list, new_statement_list=new_statement_list)
            else:
                if statement.id == statement_id:
                    is_front = False

            if is_front:
                front_st_list.append(statement)
            else:
                back_st_list.append(statement)

        # print("==========================")
        # for st in result_list:
        #     print(st.to_string())
        # print("==========================")

        if is_front is False:
            if len(front_st_list) > 0:
                result_list.extend(front_st_list)
                front_st_list.pop()

            if len(new_statement_list) > 0:
                result_list.extend(new_statement_list)

            if len(back_st_list) > 0:
                back_st_list.pop(0)
                result_list.extend(back_st_list)
        else:
            result_list.extend(front_st_list)

        # print("++++++++++++++++++++++++++")
        # for st in result_list:
        #     print(st.to_string())
        # print("++++++++++++++++++++++++++")
        return result_list

    def replace_statement(self, statement_id, new_statement_list):
        self.statement_list = self.__find_replace_st_list(
            statement_id=statement_id,
            statement_list=self.statement_list,
            new_statement_list=new_statement_list
        )
    def insert_statement_list(self, statement_id, new_statement_list):
        self.statement_list = self.__insert_st_list(
            statement_id=statement_id,
            statement_list=self.statement_list,
            new_statement_list=new_statement_list
        )

    def __find_replace_param(self, new_param, old_param, statement_list):
        result_list = []
        old_st_list = copy.deepcopy(statement_list)
        for statement in old_st_list:
            if type(statement) == SRIFStatement:
                statement.pos_statement_list = self.__find_replace_param(
                    new_param=new_param,
                    old_param=old_param,
                    statement_list=statement.pos_statement_list
                )
                statement.neg_statement_list = self.__find_replace_param(
                    new_param=new_param,
                    old_param=old_param,
                    statement_list=statement.neg_statement_list
                )
            elif type(statement) == SRSwitchStatement:
                for switch_case in statement.switch_case_list:
                    switch_case.statement_list = self.__find_replace_param(
                        new_param=new_param,
                        old_param=old_param,
                        statement_list=switch_case.statement_list
                    )
            elif type(statement) == SRFORStatement:
                statement.child_statement_list = self.__find_replace_param(
                    new_param=new_param,
                    old_param=old_param,
                    statement_list=statement.child_statement_list
                )
            elif type(statement) == SRWhileStatement:
                statement.child_statement_list = self.__find_replace_param(
                    new_param=new_param,
                    old_param=old_param,
                    statement_list=statement.child_statement_list
                )
            elif type(statement) == SRTRYStatement:
                statement.try_statement_list = self.__find_replace_param(
                    new_param=new_param,
                    old_param=old_param,
                    statement_list=statement.try_statement_list
                )
                for cb in statement.catch_block_list:
                    cb.child_statement_list = self.__find_replace_param(
                        new_param=new_param,
                        old_param=old_param,
                        statement_list=cb.child_statement_list
                    )
                statement.final_block_statement_list = self.__find_replace_param(
                    new_param=new_param,
                    old_param=old_param,
                    statement_list=statement.final_block_statement_list
                )
            else:
                statement.replace_param(
                    new_param=new_param,
                    old_param=old_param)
            statement.replace_param(
                new_param=new_param,
                old_param=old_param)
            result_list.append(statement)
        return result_list

    def replace_all_param(self, old_param_list, new_param_list, statement_list):
        result = []
        st_list = statement_list
        if len(new_param_list) == 0:
            return st_list
        for index, np in enumerate(new_param_list):
            if index < len(old_param_list):
                print(old_param_list[index].name)
                if np != old_param_list[index].name:
                    npw = [old_param_list[index].to_string(), "=", np, ";"]
                else:
                    npw = [old_param_list[index].name, "=", np, ";"]
                ns = SRStatement(
                    id="n1"+str(index),
                    word_list=npw,
                    type="fake"
                )
                result.append(ns)
        result.extend(st_list)
        # for index, np in enumerate(new_param_list):
        #     if index < len(old_param_list):
        #         result = self.__find_replace_param(
        #             new_param=np,
        #             old_param=old_param_list[index],
        #             statement_list=st_list
        #         )
        #         st_list = result
        return result

    def replace_all_var(self, old_var_list, new_var_list, statement_list):
        result = []
        st_list = self.statement_list

        if len(new_var_list) == 0:
            return st_list
        for index, np in enumerate(new_var_list):
            # if index < len(old_var_list):
            # print(np)
            result = self.__find_replace_param(
                new_param=np,
                old_param=old_var_list[index],
                statement_list=st_list
            )
            st_list = result
        return result

    def __replace_method_with_var(self, method_name, var_name, statement_list):
        result_list = []
        old_st_list = copy.deepcopy(statement_list)
        for statement in old_st_list:
            if type(statement) == SRIFStatement:
                statement.pos_statement_list = self.__replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name,
                    statement_list=statement.pos_statement_list
                )
                statement.neg_statement_list = self.__replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name,
                    statement_list=statement.neg_statement_list
                )
            elif type(statement) == SRFORStatement:
                statement.child_statement_list = self.__replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name,
                    statement_list=statement.child_statement_list
                )
            elif type(statement) == SRWhileStatement:
                statement.child_statement_list = self.__replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name,
                    statement_list=statement.child_statement_list
                )
            elif type(statement) == SRTRYStatement:
                statement.try_statement_list = self.__replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name,
                    statement_list=statement.try_statement_list
                )
                for cb in statement.catch_block_list:
                    cb.child_statement_list = self.__replace_method_with_var(
                        method_name=method_name,
                        var_name=var_name,
                        statement_list=cb.child_statement_list
                    )
                statement.final_block_statement_list = self.__replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name,
                    statement_list=statement.final_block_statement_list
                )
            else:
                statement.replace_method_with_var(
                    method_name=method_name,
                    var_name=var_name)
            statement.replace_method_with_var(
                method_name=method_name,
                var_name=var_name)
            result_list.append(statement)
        return result_list

    def replace_method_with_var(self, method_name, var_name):
        result = []
        if len(self.statement_list) > 0:
            result = self.__replace_method_with_var(method_name=method_name,
                                                    var_name=var_name, statement_list=self.statement_list)

        return result

    def __find_replace_return(self, statement_list, l_s):
        result_list = []
        old_st_list = copy.deepcopy(statement_list)
        for statement in old_st_list:
            if type(statement) == SRIFStatement:
                statement.pos_statement_list = self.__find_replace_return(
                    statement_list=statement.pos_statement_list,
                    l_s=l_s
                )
                statement.neg_statement_list = self.__find_replace_return(
                    statement_list=statement.neg_statement_list,
                    l_s=l_s
                )
            elif type(statement) == SRSwitchStatement:
                for sc in statement.switch_case_list:
                    sc.statement_list = self.__find_replace_return(
                        statement_list=sc.statement_list,
                        l_s=l_s
                    )
            elif type(statement) == SRFORStatement:
                statement.child_statement_list = self.__find_replace_return(
                    statement_list=statement.child_statement_list,
                    l_s=l_s
                )
            elif type(statement) == SRWhileStatement:
                statement.child_statement_list = self.__find_replace_return(
                    statement_list=statement.child_statement_list,
                    l_s=l_s
                )
            elif type(statement) == SRTRYStatement:
                statement.try_statement_list = self.__find_replace_return(
                    statement_list=statement.try_statement_list,
                    l_s=l_s
                )
                for cb in statement.catch_block_list:
                    cb.child_statement_list = self.__find_replace_return(
                        statement_list=cb.child_statement_list,
                        l_s=l_s
                    )
                statement.final_block_statement_list = self.__find_replace_return(
                    statement_list=statement.final_block_statement_list,
                    l_s=l_s
                )
            else:
                statement.replace_return(l_s)

            result_list.append(statement)
        return result_list

    def replace_return_statement(self, l_s, statement_list):
        result = self.__find_replace_return(
            statement_list=statement_list,
            l_s=l_s
        )
        return result

    def find_keyword(self, keyword):
        all_statement_list = self.__get_all_statement(statement_list=self.statement_list,exclude_special=False)
        for statement in all_statement_list:
            if statement.find_keyword(keyword):
                return True
        return False

    def __refresh_sid(self, current_id, st_list):
        cid = current_id
        for statement in st_list:
            if type(statement) == SRIFStatement:
                statement.sid = cid
                cid += 1
                cid = self.__refresh_sid(cid, statement.pos_statement_list)
                cid = self.__refresh_sid(cid, statement.neg_statement_list)
            elif type(statement) == SRSwitchStatement:
                for sc in statement.switch_case_list:
                    cid += 1
                    cid = self.__refresh_sid(cid, sc.statement_list)
            elif type(statement) == SRFORStatement:
                statement.sid = cid
                cid += 1
                cid = self.__refresh_sid(cid, statement.child_statement_list)
            elif type(statement) == SRWhileStatement:
                statement.sid = cid
                cid += 1
                cid = self.__refresh_sid(cid, statement.child_statement_list)
            elif type(statement) == SRTRYStatement:
                statement.sid = cid
                cid += 1
                cid = self.__refresh_sid(cid, statement.try_statement_list)
                for cb in statement.catch_block_list:
                    cid = self.__refresh_sid(cid, cb.child_statement_list)
                if len(statement.final_block_statement_list) > 0:
                    cid = self.__refresh_sid(cid, statement.final_block_statement_list)
            else:
                statement.sid = cid
                cid += 1
        return cid
    def refresh_sid(self):
        self.__refresh_sid(0, self.statement_list)

    def to_block_string(self, statement_list, space=1):
        block_data = []
        space += 1
        for statement in statement_list:
            if type(statement) == SRIFStatement:
                s_str = ""
                for x in range(0, space):
                    s_str += "    "
                s_str += statement.to_node_string()
                s_str += "{"
                s_td = {
                    "str": s_str,
                    "sid": statement.sid
                }
                block_data.append(s_td)
                if len(statement.pos_statement_list) > 0:
                    pos_block_data = self.to_block_string(statement_list=statement.pos_statement_list, space=space)
                    block_data.extend(pos_block_data)

                if len(statement.neg_statement_list) > 0:
                    el_str = ""
                    for x in range(0, space):
                        el_str += "    "
                    el_str += "} else {"

                    el_td = {
                        "str": el_str,
                        "sid": ""
                    }
                    block_data.append(el_td)
                    neg_block_data = self.to_block_string(statement_list=statement.neg_statement_list, space=space)
                    block_data.extend(neg_block_data)

                e_str = ""
                for x in range(0, space):
                    e_str += "    "
                e_str += "}"

                e_td = {
                    "str": e_str,
                    "sid": ""
                }
                block_data.append(e_td)

            elif type(statement) == SRSwitchStatement:
                s_str = ""
                for x in range(0, space):
                    s_str += "    "
                s_str += statement.to_node_string()
                s_str += "{"
                s_td = {
                    "str": s_str,
                    "sid": statement.sid
                }
                block_data.append(s_td)

                for cb in statement.switch_case_list:
                    cb_str = ""
                    for x in range(0, space):
                        cb_str += "    "
                    cb_str += cb.to_node_string()
                    cb_td = {
                        "str": cb_str,
                        "sid": statement.sid
                    }
                    block_data.append(cb_td)

                    if len(cb.statement_list) > 0:
                        child_block_data = self.to_block_string(statement_list=cb.statement_list,
                                                                space=space)
                        block_data.extend(child_block_data)
                    cb_str = ""
                    cbe_td = {
                        "str": cb_str,
                        "sid": statement.sid
                    }
                    block_data.append(cbe_td)


                e_str = ""
                for x in range(0, space):
                    e_str += "    "
                e_str += "}"

                e_td = {
                    "str": e_str,
                    "sid": ""
                }
                block_data.append(e_td)

            elif type(statement) == SRFORStatement:
                s_str = ""
                for x in range(0, space):
                    s_str += "    "
                s_str += statement.to_node_string()
                s_str += "{"
                s_td = {
                    "str": s_str,
                    "sid": statement.sid
                }
                block_data.append(s_td)
                if len(statement.child_statement_list) > 0:
                    child_block_data = self.to_block_string(statement_list=statement.child_statement_list, space=space)
                    block_data.extend(child_block_data)
                e_str = ""
                for x in range(0, space):
                    e_str += "    "
                e_str += "}"

                e_td = {
                    "str": e_str,
                    "sid": ""
                }
                block_data.append(e_td)


            elif type(statement) == SRWhileStatement:
                s_str = ""
                for x in range(0, space):
                    s_str += "    "
                s_str += statement.to_node_string()
                s_str += "{"
                s_td = {
                    "str": s_str,
                    "sid": statement.sid
                }
                block_data.append(s_td)
                if len(statement.child_statement_list) > 0:
                    child_block_data = self.to_block_string(statement_list=statement.child_statement_list, space=space)
                    block_data.extend(child_block_data)
                e_str = ""
                for x in range(0, space):
                    e_str += "    "
                e_str += "}"

                e_td = {
                    "str": e_str,
                    "sid": ""
                }
                block_data.append(e_td)
            elif type(statement) == SRTRYStatement:
                s_str = ""
                for x in range(0, space):
                    s_str += "    "
                s_str += "try"
                s_str += "{"
                s_td = {
                    "str": s_str,
                    "sid": statement.sid
                }
                block_data.append(s_td)
                if len(statement.try_statement_list) > 0:
                    try_block_data = self.to_block_string(statement_list=statement.try_statement_list, space=space)
                    block_data.extend(try_block_data)

                es_str = ""
                for x in range(0, space):
                    es_str += "    "
                es_str += "}"
                es_td = {
                    "str": es_str,
                    "sid": ""
                }
                block_data.append(es_td)

                for cb in statement.catch_block_list:
                    cb_str = ""
                    for x in range(0, space):
                        cb_str += "    "
                    cb_str += cb.to_node_string()
                    cb_str += " {"
                    cb_td = {
                        "str": cb_str,
                        "sid": statement.sid
                    }
                    block_data.append(cb_td)

                    if len(cb.child_statement_list) > 0:
                        child_block_data = self.to_block_string(statement_list=cb.child_statement_list,
                                                                space=space)
                        block_data.extend(child_block_data)
                    cb_str = ""
                    for x in range(0, space):
                        cb_str += "    "
                    cb_str += "}"
                    cbe_td = {
                        "str": cb_str,
                        "sid": statement.sid
                    }
                    block_data.append(cbe_td)

                if len(statement.final_block_statement_list) > 0:
                    fn_str = ""
                    for x in range(0, space):
                        fn_str += "    "
                    fn_str += "finally {"
                    fn_td = {
                        "str": fn_str,
                        "sid": statement.sid
                    }
                    block_data.append(fn_td)

                    fn_block_data = self.to_block_string(statement_list=statement.final_block_statement_list, space=space)
                    block_data.extend(fn_block_data)

                    efn_str = ""
                    for x in range(0, space):
                        efn_str += "    "
                    efn_str += "}"
                    efn_td = {
                        "str": efn_str,
                        "sid": statement.sid
                    }
                    block_data.append(efn_td)

            else:
                s_str = ""
                for x in range(0, space):
                    s_str += "    "
                s_str += statement.to_node_string()
                s_td = {
                    "str": s_str,
                    "sid": statement.sid
                }
                block_data.append(s_td)
        return block_data

    def to_string_table(self, space=1):
        table_data = []
        m_str = ""
        m_str += " ".join(self.modifiers)
        m_str += " "
        m_str += self.return_type
        m_str += " "
        m_str += self.method_name

        m_str += "("
        if len(self.param_list) > 0:
            m_str += ",".join(
                map(lambda p: p.to_string(), self.param_list)
            )
        m_str += ")"
        m_str += " "
        m_str += " ".join(self.throws)
        m_str += "{"
        # m_str += "\n"
        m_td = {
            "str": m_str,
            "sid": ""
        }
        table_data.append(m_td)
        self.refresh_sid()
        block_data = self.to_block_string(statement_list=self.statement_list, space=space)
        table_data.extend(block_data)



        # if len(self.statement_list) > 0:
        #     for index, statement in enumerate(self.statement_list):
        #         s_str = ""
        #         for x in range(0, space):
        #             s_str += "    "
        #         s_str += statement.to_node_string(space=space)
        #         s_td = {
        #             "str": s_str,
        #             "sid": str(index)
        #         }
        #         table_data.append(s_td)

        e_str = ""
        for x in range(0, space - 1):
            e_str += "    "
        e_str += "}"
        e_std = {
            "str": e_str,
            "sid": ""
        }
        table_data.append(e_std)
        return table_data


    def to_json(self):
        info = {}
        info["id"] = self.id
        info["method_name"] = self.method_name
        info["return_type"] = self.return_type
        info["modifiers"] = " ".join(self.modifiers)
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info["id"] = self.id
        info["method_name"] = self.method_name
        info["return_type"] = self.return_type
        info["modifiers"] = " ".join(self.modifiers)
        return info


    def rebuild_mkg(self):
        self._rebuild_mkg(self.statement_list)

    def _rebuild_mkg(self, statement_list):
        for statement in statement_list:
            if type(statement) == SRIFStatement:
                self._rebuild_mkg(statement.pos_statement_list)
                self._rebuild_mkg(statement.neg_statement_list)
                self.fetch_common_assignment_var(statement.pos_statement_list, statement.neg_statement_list)
            elif type(statement) == SRFORStatement:
                self._rebuild_mkg(statement.child_statement_list)
            elif type(statement) == SRWhileStatement:
                self._rebuild_mkg(statement.child_statement_list)
            elif type(statement) == SRTRYStatement:
                self._rebuild_mkg(statement.try_statement_list)

    def fetch_common_assignment_var(self, stl1, stl2):
        stl1_var =[]
        stl1_ass_var = []
        stl2_var = []
        stl2_ass_var = []
        for st in stl1:
            stl1_var.extend(st.var)
            stl1_ass_var.extend(st.assignment_var)

        for st in stl2:
            stl2_var.extend(st.var)
            stl2_ass_var.extend(st.assignment_var)

        stl1_ass_var.reverse()
        stl2_ass_var.reverse()

        for v in stl2_var:
            if v in stl1_var:
                for asv2 in stl2_ass_var:
                    edge = self.mkg.find_edge(asv2.label, v)
                    if edge is not None:
                        if edge.type == ASSIGNMENT:
                            asv1 = self.get_latest_assignment_var(stl1_ass_var, v)
                            for edge in self.mkg.edges:
                                if edge.type == DATA_DEPENDENCY:
                                    if edge.target == asv2:
                                        new_dd_edge = self.mkg.get_or_create_edge(edge.source, asv1, DATA_DEPENDENCY)

    def get_latest_assignment_var(self, asv_l, v):
        for asv in asv_l:
            edge = self.mkg.find_edge(asv.label, v)
            if edge is not None:
                if edge.type == ASSIGNMENT:
                    return edge.source

class SRConstructor(SRCore):
    def __init__(self, word_list=[], id="-1", param_list=[], name="", modifiers="", statement_list=[]):
        self.word_list = word_list
        # self.method_name = method_name
        self.statement_list = statement_list
        self.param_list = param_list

        self.id = id
        self.modifiers = modifiers

        self.comment = ""

        self.mkg = None

    def to_string(self, space=1):
        return " ".join(self.word_list)


class SRParam:
    def __init__(self, type, name, dimensions=None):
        self.type = type
        self.name = name
        self.dimensions = dimensions

    def to_string(self):
        if self.dimensions is not None:
            return self.type + " " + self.name + " " + self.dimensions
        return self.type + " " + self.name