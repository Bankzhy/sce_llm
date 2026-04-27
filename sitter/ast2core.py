from tree_sitter import Language, Parser
import os
import re
import uuid

from tree_sitter_languages import get_parser

from reflect.sr_field import SRField
from reflect.sr_method import SRMethod, SRParam, SRConstructor
from reflect.sr_program import SRProgram
from reflect.sr_project import SRProject
from reflect.sr_statement import SRStatement, SRFORStatement, SRIFStatement, SRWhileStatement, SRTRYStatement, \
    CatchBlock, SRSwitchStatement, SRSwitchCase, MethodInvoke

from reflect.sr_class import SRClass

class ASTParse:
    def __init__(self, project_path, language):
        self.PROGRAM = "program"
        self.PACKAGE_DECLARATION = "package_declaration"
        self.IMPORT_DECLARATION = "import_declaration"
        self.BLOCK_COMMENT = "block_comment"
        self.CLASS_DECLARATION = "class_declaration"
        self.CLASS_DEFINITION = 'class_definition'
        self.CLASS_BODY = "class_body"
        self.FIELD_DECLARATION = "field_declaration"
        self.CONSTRUCTOR_DECLARATION = "constructor_declaration"
        self.METHOD_DECLARATION = "method_declaration"
        self.FUNCTION_DEFINITION = "function_definition"
        self.BLOCK = "block"
        self.FOR_STATEMENT = "for_statement"
        self.ENHANCED_FOR_STATEMENT = "enhanced_for_statement"
        self.MODIFIERS = "modifiers"
        self.IDENTIFIER = "identifier"
        self.TYPE_IDENTIFIER = "type_identifier"
        self.SUPERCLASS = "superclass"
        self.VARIABLE_DECLARATOR = "variable_declarator"
        self.FORMAL_PARAMETERS = "formal_parameters"
        self.FORMAL_PARAMETER = "formal_parameter"
        self.BLOCK_START = "{"
        self.BLOCK_END = "}"
        self.EXPRESSION_STATEMENT = "expression_statement"
        self.RETURN_STATEMENT = "return_statement"
        self.LOCAL_VARIABLE_DECLARATION = "local_variable_declaration"
        self.IF_STATEMENT = "if_statement"
        self.WHILE_STATEMENT = "while_statement"
        self.ASSIGNMENT_EXPRESSION = "assignment_expression"
        self.BINARY_EXPRESSION = "binary_expression"
        self.UPDATE_EXPRESSION = "update_expression"
        self.PARENTHESIZED_EXPRESSION = "parenthesized_expression"
        self.CONSTRUCTOR_BODY = "constructor_body"
        self.TRY_STATEMENT = "try_statement"
        self.CATCH_CLAUSE = "catch_clause"
        self.FINALLY_CLAUSE = "finally_clause"
        self.CATCH_FORMAL_PARAMETER = "catch_formal_parameter"
        self.LINE_COMMENT = "line_comment"
        self.THROWS = "throws"
        self.SWITCH_EXPRESSION = "switch_expression"
        self.PARENTHESIZED_EXPRESSION = "parenthesized_expression"
        self.SWITCH_BLOCK = "switch_block"
        self.SWITCH_BLOCK_STATEMENT_GROUP = "switch_block_statement_group"
        self.SWITCH_LABEL = "switch_label"
        self.LABELED_STATEMENT = "labeled_statement"
        self.LOCAL_VARIABLE_DECLARATION = "local_variable_declaration"
        self.EXPRESSION_STATEMENT = "expression_statement"
        self.IDENTIFIER = "identifier"
        self.METHOD_INVOCATION = "method_invocation"
        self.ARGUMENT_LIST = "argument_list"
        self.symbol = ["[","]","<",">","{","}",",","(",")"]

        self.language = language
        self.JAVA_LANGUAGE = None
        self.java_lib_path = None
        self.so_gen_path = None
        self.parse = None
        self.fileList = []
        self.project_path = project_path
        self.sr_project = SRProject(
            project_name=self.project_path,
            project_id=self.get_uuid()
        )

    def setup(self):
        # self.so_gen_path = './build/java-languages.so'
        # self.java_lib_path = 'C:\worksapce\example\\tree-sitter-java'
        # Language.build_library(
        #     # Store the library in the `build` directory
        #     self.so_gen_path,
        #
        #     # Include one or more languages
        #     [
        #         self.java_lib_path,
        #     ]
        # )
        #
        # self.JAVA_LANGUAGE = Language(self.so_gen_path, 'java')
        # self.parse = Parser()
        # self.parse.set_language(self.JAVA_LANGUAGE)
        # getf = lambda self:get_parser()
        # self.parse = self.function()
        # get_parser("java")
        self.parse = get_parser(self.language)

        # self.parser.set_language(language)

    def function(self):
        get_parser("java")

    # Generate UUID for element
    def get_uuid(self):
        return uuid.uuid1().hex
    # Function can get *.xls/*.xlsx file from the directory
    """
    dirpath: str, the path of the directory
    """
    def getfiles(self, dirPath):
        # open directory
        files = os.listdir(dirPath)
        # re match *.xls/xlsx，you can change 'xlsx' to 'doc' or other file types.
        ptn = re.compile('.*\.'+self.language)
        for f in files:
            # isdir, call self
            if (os.path.isdir(dirPath / f)):
                self.getfiles(dirPath / f)
            # isfile, judge
            elif (os.path.isfile(dirPath / f)):
                res = ptn.match(f)
                # print(res)
                if (res != None):
                    self.fileList.append(dirPath / res.group())
            else:
                self.fileList.append(dirPath + '\\无效文件')

    def parse_field_node(self, root_node):
        new_sr_field = SRField(
            id=self.get_uuid()
        )
        new_sr_field.start_line = (root_node.start_point[0] + 1)
        new_sr_field.end_line = (root_node.end_point[0] + 1)
        word_list = []

        for node in root_node.children:
            word_list.append(node.text.decode())
            if node.type == self.MODIFIERS:
                new_sr_field.modifiers = node.text.decode()
            elif node.type.startswith("type_") or node.type.endswith("type"):
                new_sr_field.field_type = node.text.decode()
            elif node.type == self.VARIABLE_DECLARATOR:
                if node.child_count == 1:
                    new_sr_field.field_name = node.text.decode()
                elif node.child_count == 3:
                    new_sr_field.field_value = node.children[2].text.decode()
                    new_sr_field.field_name = node.children[0].text.decode()

        new_sr_field.word_list = word_list
        return new_sr_field

    def parse_param(self, root_node):
        param_list = []
        for node in root_node.children:
            if node.type == self.FORMAL_PARAMETER:
                if len(node.children) == 3:
                    new_sr_param = SRParam(
                        type=node.children[0].text.decode(),
                        name=node.children[1].text.decode(),
                        dimensions=node.children[2].text.decode()
                    )
                else:
                    new_sr_param = SRParam(
                        type=node.children[0].text.decode(),
                        name=node.children[1].text.decode()
                    )
                param_list.append(new_sr_param)
        return param_list

    def parse_enhanced_for_statement(self, root_node):
        new_sr_for_statement = SRFORStatement(
            id=self.get_uuid()
        )
        new_sr_for_statement.start_line = (root_node.start_point[0] + 1)
        new_sr_for_statement.end_line = (root_node.end_point[0] + 1)
        new_sr_for_statement.end_condition = []
        word_list = self.statement_node_to_word_list(root_node)
        local_word_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())

            if node.type != self.BLOCK:
                result = self.statement_node_to_word_list(node)
                local_word_list.extend(result)

            if node.type == self.TYPE_IDENTIFIER or node.type == self.IDENTIFIER or node.type == ":":
                new_sr_for_statement.end_condition.append(node.text.decode())
            elif node.type == self.BLOCK:
                child_statement_list = self.parse_block(node)

                new_sr_for_statement.child_statement_list = child_statement_list
            else:
                children = [node]
                fake_node = FakeNode(
                    children=children
                )
                child_statement_list = self.parse_block(fake_node)
                new_sr_for_statement.child_statement_list = child_statement_list

        new_sr_for_statement.word_list = word_list
        new_sr_for_statement.local_word_list = local_word_list
        return new_sr_for_statement

    def parse_for_statement(self, root_node):
        new_sr_for_statement = SRFORStatement(
            id=self.get_uuid()
        )
        new_sr_for_statement.start_line = (root_node.start_point[0] + 1)
        new_sr_for_statement.end_line = (root_node.end_point[0] + 1)
        word_list = self.statement_node_to_word_list(root_node)
        local_word_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type != self.BLOCK:
                result = self.statement_node_to_word_list(node)
                local_word_list.extend(result)

            if node.type == self.LOCAL_VARIABLE_DECLARATION or node.type == self.ASSIGNMENT_EXPRESSION:
                # new_sr_for_statement.init = list(map(lambda n: n.text.decode(), node.children))
                new_sr_for_statement.init = self.statement_node_to_word_list(node)
            elif node.type == self.BINARY_EXPRESSION:
                # new_sr_for_statement.end_condition = list(map(lambda n: n.text.decode(), node.children))
                new_sr_for_statement.end_condition = self.statement_node_to_word_list(node)
            elif node.type == self.UPDATE_EXPRESSION:
                # new_sr_for_statement.update = list(map(lambda n: n.text.decode(), node.children))
                new_sr_for_statement.update = self.statement_node_to_word_list(node)

            elif node.type == self.BLOCK:
                child_statement_list = self.parse_block(node)

                new_sr_for_statement.child_statement_list = child_statement_list
            else:
                children = [node]
                fake_node = FakeNode(
                    children=children
                )
                child_statement_list = self.parse_block(fake_node)
                new_sr_for_statement.child_statement_list = child_statement_list

        new_sr_for_statement.word_list = word_list
        new_sr_for_statement.local_word_list = local_word_list
        return new_sr_for_statement
    def parse_catch_block(self, root_node):
        new_catch_block = CatchBlock()
        for node in root_node.children:
            if node.type == self.CATCH_FORMAL_PARAMETER:
                new_catch_block.catch_param = node.text.decode()
                new_catch_block.word_list = self.statement_node_to_word_list(node)
            elif node.type == self.BLOCK:
                new_catch_block.child_statement_list = self.parse_block(node)
        return new_catch_block

    def parse_final_block(self, root_node):
        statement_list = []
        for node in root_node.children:
                # print("======================")
                # print(node.type)
                # print(node.text)
            if node.type == self.BLOCK:
                statement_list = self.parse_block(node)
        return statement_list

    def parse_try_statement(self, root_node):
        new_sr_try_statement = SRTRYStatement(
            id=self.get_uuid(),
        )
        new_sr_try_statement.start_line = (root_node.start_point[0] + 1)
        new_sr_try_statement.end_line = (root_node.end_point[0] + 1)
        word_list = []
        new_catch_block_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.BLOCK:
                new_sr_try_statement.try_statement_list = self.parse_block(node)
            elif node.type == self.CATCH_CLAUSE:
                new_catch_block = self.parse_catch_block(node)
                new_catch_block_list.append(new_catch_block)
            elif node.type == self.FINALLY_CLAUSE:
                new_sr_try_statement.final_block_statement_list = self.parse_final_block(node)

        new_sr_try_statement.catch_block_list = new_catch_block_list
        word_list.append("try-catch: hasError?")
        new_sr_try_statement.word_list = word_list
        return new_sr_try_statement

    def parse_if_statement(self, root_node):
        else_index = -1
        new_sr_if_statement = SRIFStatement(
            id=self.get_uuid()
        )
        new_sr_if_statement.start_line = (root_node.start_point[0] + 1)
        new_sr_if_statement.end_line = (root_node.end_point[0] + 1)
        word_list = self.statement_node_to_word_list(root_node)
        local_word_list = []
        new_sr_if_statement.word_list = word_list
        for index, node in enumerate(root_node.children):
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            # new_sr_if_statement.word_list.append(node.text.decode())
            if node.type != self.BLOCK:
                result = self.statement_node_to_word_list(node)
                local_word_list.extend(result)

            if node.type == self.PARENTHESIZED_EXPRESSION:
                new_sr_if_statement.condition = self.statement_node_to_word_list(node)
                # new_sr_if_statement.word_list.append("if")
                # new_sr_if_statement.word_list.extend(new_sr_if_statement.condition)
            elif node.type == "else":
                else_index = index
        if else_index != -1:
            pos_block_node = root_node.children[else_index-1]
            neg_block_node = root_node.children[else_index+1]
            if pos_block_node.type == self.BLOCK:
                new_sr_if_statement.pos_statement_list = self.parse_block(pos_block_node)
            else:
                new_sr_statement = SRStatement(
                    id=self.get_uuid(),
                    type=pos_block_node.type,
                    word_list=list(map(lambda n: n.text.decode(), pos_block_node.children))
                )
                new_sr_statement.start_line = (node.start_point[0] + 1)
                new_sr_statement.end_line = (node.end_point[0] + 1)
                new_sr_if_statement.pos_statement_list = [new_sr_statement]

            if neg_block_node.type == self.BLOCK:
                new_sr_if_statement.neg_statement_list = self.parse_block(neg_block_node)
            elif neg_block_node.type == self.IF_STATEMENT:
                new_sr_if_statement.neg_statement_list = [self.parse_if_statement(neg_block_node)]
            else:
                new_sr_statement = SRStatement(
                    id=self.get_uuid(),
                    type=neg_block_node.type,
                    word_list=list(map(lambda n: n.text.decode(), neg_block_node.children))
                )
                new_sr_statement.start_line = (node.start_point[0] + 1)
                new_sr_statement.end_line = (node.end_point[0] + 1)
                new_sr_if_statement.neg_statement_list = [new_sr_statement]
        else:
            pos_block_node = root_node.children[root_node.child_count -1]
            if pos_block_node.type == self.BLOCK:
                new_sr_if_statement.pos_statement_list = self.parse_block(pos_block_node)
            else:
                new_sr_statement = SRStatement(
                    id=self.get_uuid(),
                    type=pos_block_node.type,
                    word_list=list(map(lambda n: n.text.decode(), pos_block_node.children))
                )
                new_sr_statement.start_line = (node.start_point[0] + 1)
                new_sr_statement.end_line = (node.end_point[0] + 1)
                new_sr_if_statement.pos_statement_list = [new_sr_statement]
        new_sr_if_statement.local_word_list = local_word_list
        return new_sr_if_statement


    def parse_switch_block_group(self, root_node):
        new_sr_switch_case = SRSwitchCase(
            id=self.get_uuid()
        )

        new_sr_switch_case.word_list
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())

            if node.type == self.SWITCH_LABEL:
                new_sr_switch_case.condition = self.statement_node_to_word_list(node)
        new_sr_switch_case.statement_list = []
        new_sr_switch_case.statement_list = self.parse_block(root_node)

        return new_sr_switch_case


    def parse_switch_block(self, root_node):
        new_switch_case_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.SWITCH_BLOCK_STATEMENT_GROUP:
               new_switch_case_list.append(self.parse_switch_block_group(node))
        return new_switch_case_list

    def parse_switch_statement(self, root_node):
        new_sr_switch_statement = SRSwitchStatement(
            id=self.get_uuid()
        )
        new_sr_switch_statement.start_line = (root_node.start_point[0] + 1)
        new_sr_switch_statement.end_line = (root_node.end_point[0] + 1)
        word_list = self.statement_node_to_word_list(root_node)
        local_word_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type != self.BLOCK:
                result = self.statement_node_to_word_list(node)
                local_word_list.extend(result)
            if node.type == self.PARENTHESIZED_EXPRESSION:
                new_sr_switch_statement.condition = self.statement_node_to_word_list(node)
            elif node.type == self.SWITCH_BLOCK:
                new_sr_switch_statement.switch_case_list = self.parse_switch_block(node)
        new_sr_switch_statement.local_word_list=local_word_list
        new_sr_switch_statement.word_list = word_list
        return new_sr_switch_statement


    def parse_while_statement(self, root_node):
        new_sr_while_statement = SRWhileStatement(
            id=self.get_uuid()
        )
        new_sr_while_statement.start_line = (root_node.start_point[0] + 1)
        new_sr_while_statement.end_line = (root_node.end_point[0] + 1)
        word_list = self.statement_node_to_word_list(root_node)
        local_word_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())

            if node.type != self.BLOCK:
                result = self.statement_node_to_word_list(node)
                local_word_list.extend(result)

            if node.type == self.PARENTHESIZED_EXPRESSION:
                # new_sr_while_statement.end_condition = list(map(lambda n: n.text.decode(), node.children))
                new_sr_while_statement.end_condition = self.statement_node_to_word_list(node)
                # new_sr_while_statement.word_list.append("while")
                # new_sr_while_statement.word_list.extend(new_sr_while_statement.end_condition)
            elif node.type == self.BLOCK:
                new_sr_while_statement.child_statement_list = self.parse_block(node)
            else:
                children = [node]
                fake_node = FakeNode(
                    children=children
                )
                child_statement_list = self.parse_block(fake_node)
                new_sr_while_statement.child_statement_list = child_statement_list

        new_sr_while_statement.word_list = word_list
        new_sr_while_statement.local_word_list = local_word_list
        return new_sr_while_statement

    def parse_labeled_statement(self, root_node):
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.FOR_STATEMENT:
                return self.parse_for_statement(node)
        word_list = ["// "]
        word_list.extend(self.statement_node_to_word_list(node))
        new_sr_statement = SRStatement(
            id=self.get_uuid(),
            type=node.type,
            word_list=word_list
        )
        new_sr_statement.start_line = (node.start_point[0] + 1)
        new_sr_statement.end_line = (node.end_point[0] + 1)
        return new_sr_statement

    def parse_block(self, root_node):
        statement_list = []
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type != self.BLOCK_START and node.type != self.BLOCK_END:
                if node.type == self.FOR_STATEMENT:
                    new_sr_statement = self.parse_for_statement(node)
                    for sub_node in node.children:
                        self.parse_statement(new_sr_statement, sub_node)
                    statement_list.append(new_sr_statement)
                elif node.type == self.ENHANCED_FOR_STATEMENT:
                    new_sr_statement = self.parse_enhanced_for_statement(node)
                    for sub_node in node.children:
                        self.parse_statement(new_sr_statement, sub_node)
                    statement_list.append(new_sr_statement)
                elif node.type == self.IF_STATEMENT:
                    new_sr_statement = self.parse_if_statement(node)
                    for sub_node in node.children:
                        self.parse_statement(new_sr_statement, sub_node)
                    statement_list.append(new_sr_statement)
                elif node.type == self.WHILE_STATEMENT:
                    new_sr_statement = self.parse_while_statement(node)
                    for sub_node in node.children:
                        self.parse_statement(new_sr_statement, sub_node)
                    statement_list.append(new_sr_statement)
                elif node.type == self.TRY_STATEMENT:
                    statement_list.append(self.parse_try_statement(node))
                elif node.type == self.SWITCH_EXPRESSION:
                    statement_list.append(self.parse_switch_statement(node))
                elif node.type == self.SWITCH_LABEL:
                    continue
                elif node.type == self.LABELED_STATEMENT:
                    statement_list.append(self.parse_labeled_statement(node))
                else:
                    # word_list = list(map(lambda n: n.text.decode(), node.children))
                    if node.type != self.LINE_COMMENT:
                        word_list = self.statement_node_to_word_list(node)
                        new_sr_statement = SRStatement(
                            id=self.get_uuid(),
                            type=node.type,
                            word_list=word_list
                        )
                        self.parse_statement(new_sr_statement, node)

                        new_sr_statement.start_line = (node.start_point[0] + 1)
                        new_sr_statement.end_line = (node.end_point[0] + 1)
                        statement_list.append(new_sr_statement)
        return statement_list

    def parse_statement(self, statement, statement_node):
        if statement_node.type == self.LOCAL_VARIABLE_DECLARATION:
            statement.datatype = self.fetch_data_type(statement_node.children[0])
            statement.var, statement.value = self.fetch_var(statement_node.children[1], statement)
            statement.type = self.LOCAL_VARIABLE_DECLARATION
        elif statement_node.type == self.EXPRESSION_STATEMENT:
            if statement_node.children[0].type == self.ASSIGNMENT_EXPRESSION:
                statement.var, statement.value = self.fetch_var(statement_node.children[0], statement)
            elif statement_node.children[0].type == self.METHOD_INVOCATION:
                self.parse_method_invocation(statement_node.children[0], statement)

    def fetch_data_type(self, node):
        data_type = []
        for child in node.children:
            if len(child.children) > 0:
                for sub_child in child.children:
                    sub_child_text = sub_child.text.decode()
                    if sub_child_text not in self.symbol:
                        data_type.append(sub_child_text)
            else:
                data_type.append(child.text.decode())

        return data_type

    def fetch_var(self, node, statement):
        var = []
        value = []
        for index, child in enumerate(node.children):
            if child.type == "=":
                if node.children[index+1].type==self.METHOD_INVOCATION:
                    self.parse_method_invocation(node.children[index+1], statement)
                else:
                    value.append(node.children[index+1].text.decode())
            if child.type == self.IDENTIFIER:
                var.append(child.text.decode())
        return var, value


    def parse_method_invocation(self, statement_node, statement):
        mi = MethodInvoke()
        for index, node in enumerate(statement_node.children):
            if node.type == ".":
                mi.method_name = statement_node.children[index+1].text.decode()
                mi.parent = statement_node.children[index - 1].text.decode()
            if node.type == self.ARGUMENT_LIST:
                for sub_node in node.children:
                    if sub_node.type == self.IDENTIFIER:
                        mi.param.append(sub_node.text.decode())
        statement.method.append(mi)

    def statement_node_to_word_list(self, root_node):
        result_list = []
        if root_node.child_count > 0:
            for node in root_node.children:
                if node.child_count > 0:
                    result_list.extend(self.statement_node_to_word_list(node))
                else:
                    result_list.append(node.text.decode())
        else:
            result_list.append(root_node.text.decode())
        return result_list


    def parse_method_node(self, root_node):
        new_sr_method = SRMethod(
            id=self.get_uuid()
        )
        new_sr_method.start_line = (root_node.start_point[0] + 1)
        new_sr_method.end_line = (root_node.end_point[0] + 1)
        word_list = []
        word_list.extend(self.statement_node_to_word_list(root_node))
        new_sr_method.text = root_node.text.decode()
        for node in root_node.children:
            # print("++++++++++++++++++++++++++++++++++++")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.MODIFIERS:
                new_sr_method.modifiers = self.statement_node_to_word_list(node)
            elif node.type.startswith("type_") or node.type.endswith("type"):
                new_sr_method.return_type = node.text.decode()
            elif node.type == self.IDENTIFIER:
                new_sr_method.method_name = node.text.decode()
            elif node.type == self.FORMAL_PARAMETERS:
                new_sr_method.param_list = self.parse_param(node)
            elif node.type == self.BLOCK:
                new_sr_method.statement_list = self.parse_block(node)
            elif node.type == self.THROWS:
                new_sr_method.throws = self.statement_node_to_word_list(node)

        new_sr_method.word_list = word_list
        return new_sr_method

    def parse_constructor(self, root_node):
        new_sr_constructor = SRConstructor(
            id=self.get_uuid()
        )
        new_sr_constructor.start_line = (root_node.start_point[0] + 1)
        new_sr_constructor.end_line = (root_node.end_point[0] + 1)
        word_list = []
        for node in root_node.children:
            word_list.append(node.text.decode())
            # print("++++++++++++++++++++++++++++++++++++")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.MODIFIERS:
                new_sr_constructor.modifiers = node.text.decode()
            elif node.type == self.IDENTIFIER:
                new_sr_constructor.name = node.text.decode()
            elif node.type == self.FORMAL_PARAMETERS:
                new_sr_constructor.param_list = self.parse_param(node)
            elif node.type == self.CONSTRUCTOR_BODY:
                new_sr_constructor.statement_list = self.parse_block(node)
        new_sr_constructor.word_list = word_list
        return new_sr_constructor

    def parse_class_block(self, root_node):
        field_list = []
        method_list = []
        constructor_list = []
        for index, node in enumerate(root_node.children):
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.FIELD_DECLARATION:
                new_sr_field = self.parse_field_node(node)
                field_list.append(new_sr_field)
            elif node.type == self.METHOD_DECLARATION:
                new_sr_method = self.parse_method_node(node)
                if index != 0:
                    if "comment" in root_node.children[index - 1].type:
                        new_sr_method.comment = root_node.children[index - 1].text.decode()
                method_list.append(new_sr_method)
            elif node.type == self.CONSTRUCTOR_DECLARATION:
                constructor_list.append(self.parse_constructor(node))
            elif node.type == self.FUNCTION_DEFINITION:
                new_sr_method = self.parse_method_node(node)
                if index != 0:
                    if "comment" in root_node.children[index - 1].type:
                        new_sr_method.comment = root_node.children[index - 1].text.decode()
                method_list.append(new_sr_method)
        return field_list, method_list, constructor_list

    def parse_class_node(self, root_node):
        new_sr_class = SRClass(id=self.get_uuid())
        new_sr_class.start_line = (root_node.start_point[0] + 1)
        new_sr_class.end_line = (root_node.end_point[0] + 1)
        for node in root_node.children:
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.MODIFIERS:
                new_sr_class.modifiers = node.text.decode()
            elif node.type == self.IDENTIFIER:
                new_sr_class.class_name = node.text.decode()
            elif node.type == self.SUPERCLASS:
                extend_l = self.statement_node_to_word_list(node)
                new_sr_class.extends = extend_l
            elif node.type == self.CLASS_BODY:
                field_list, method_list, constructor_list = self.parse_class_block(node)
                new_sr_class.field_list = field_list
                new_sr_class.method_list = method_list
                new_sr_class.constructor_list = constructor_list
            elif node.type == self.BLOCK:
                field_list, method_list, constructor_list = self.parse_class_block(node)
                new_sr_class.field_list = field_list
                new_sr_class.method_list = method_list
                new_sr_class.constructor_list = constructor_list

        return new_sr_class

    def parse_program_node(self, root_node, program_name):
        class_list = []
        import_list = []
        package_name = ''
        for index, node in enumerate(root_node.children):
            # print("======================")
            # print(node.type)
            # print(node.text.decode())
            if node.type == self.PACKAGE_DECLARATION:
                package_name = node.text.decode()
            elif node.type == self.IMPORT_DECLARATION:
                import_list.append(node.text.decode())
            elif node.type == self.CLASS_DECLARATION:
                new_sr_class = self.parse_class_node(node)
                new_sr_class.import_list = import_list
                new_sr_class.package_name = package_name

                if index != 0:
                    if "comment" in root_node.children[index - 1].type:
                        new_sr_class.comment = root_node.children[index - 1].text.decode()
                class_list.append(new_sr_class)
            elif node.type == self.CLASS_DEFINITION:
                new_sr_class = self.parse_class_node(node)
                new_sr_class.import_list = import_list
                new_sr_class.package_name = package_name

                if index != 0:
                    if "comment" in root_node.children[index - 1].type:
                        new_sr_class.comment = root_node.children[index - 1].text.decode()
                class_list.append(new_sr_class)
        # program_name_list = program_name.split("\\")
        # program_name = program_name_list[len(program_name_list)-1]
        new_sr_program = SRProgram(
            program_name=program_name,
            program_id=self.get_uuid(),
            class_list=class_list,
            import_list=import_list,
            package_name=package_name,
        )
        new_sr_program.start_line = (root_node.start_point[0] + 1)
        new_sr_program.end_line = (root_node.end_point[0] + 1)
        return new_sr_program

    def do_parse(self):
        self.getfiles(self.project_path)
        program_list = []
        new_sr_project = SRProject(
            project_id=self.get_uuid(),
            project_name=self.project_path
        )
        for file_path in self.fileList:
            try:
                file = open(file_path, encoding='utf-8')
                file_content = file.read()
                tree = self.parse.parse(bytes(file_content, "utf8"))
                program = self.parse_program_node(tree.root_node, file_path)
                program_list.append(program)
            except Exception as e:
                print(e)
                continue
        new_sr_project.program_list = program_list
        return new_sr_project

    def do_parse_one_file(self, file_path):
        program_list = []
        new_sr_project = SRProject(
            project_id=self.get_uuid(),
            project_name=self.project_path
        )

        try:
            file = open(file_path, encoding='utf-8')
            file_content = file.read()
            tree = self.parse.parse(file_content.encode())
            program = self.parse_program_node(tree.root_node, file_path)
            program_list.append(program)
        except Exception as e:
            print(e)

        new_sr_project.program_list = program_list
        return new_sr_project

    def do_parse_content(self, content):
        program_list = []
        new_sr_project = SRProject(
            project_id=self.get_uuid(),
            project_name=self.project_path
        )

        # try:
        tree = self.parse.parse(content.encode())
        program = self.parse_program_node(tree.root_node, "")
        program_list.append(program)
        # except Exception as e:
        #     print(e)

        new_sr_project.program_list = program_list
        return new_sr_project

class FakeNode():
    start_point = (0,0)
    end_point = (0,0)
    def __init__(self, children):
        self.children = children
