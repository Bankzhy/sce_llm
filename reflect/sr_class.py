import json

from reflect.sr_core import SRCore


class SRClass(SRCore):
    def __init__(self, id="", class_name="", modifiers="",
                 implement_list=[], extends=[], field_list=[], method_list=[], constructor_list=[], import_list=[], package_name=""):
        self.class_name = class_name
        self.implement_list = implement_list
        self.extends = extends
        self.field_list = field_list
        self.method_list = method_list
        self.id = id
        self.modifiers = modifiers
        self.constructor_list = constructor_list
        self.import_list = import_list
        self.package_name = package_name
        self.comment = ""

    def to_string(self, space=1):
        result = ""
        # result += self.packddage_name
        # result += "\n"

        for ip in self.import_list:
            result += ip
            result += "\n"
        result += "\n"

        if self.modifiers != "":
            result += self.modifiers
        result += " "

        result += "class"
        result += " "
        result += self.class_name
        result += " "

        # result += "extends"
        if len(self.extends) > 0:
            for es in self.extends:
                result += es
                result += " "


        if len(self.implement_list) > 0:
            result += "implements"
            result += ",".join(self.implement_list)

        result += " {"
        result += "\n"

        for field in self.field_list:
            for x in range(0, space):
                result += "    "
            result += field.to_string()
            result += "\n"

        result += "\n"

        if len(self.constructor_list) > 0:
            for constructor in self.constructor_list:
                for x in range(0, space):
                    result += "    "
                space = 1
                result += constructor.to_string(space=space)
                result += "\n"
            result += "\n"
            result += "\n"

        for method in self.method_list:
            for x in range(0, space):
                result += "    "
            space = 1
            result += method.to_string(space=space)
            result += "\n"
            result += "\n"
        result += "}"
        result += "\n"

        return result

    def to_json(self):
        info = {}
        info["className"] = self.class_name
        info["extends"] = self.extends
        info["id"] = self.id
        info["fields"] = list(map(lambda x: x.to_json(), self.field_list))
        info["methods"] = list(map(lambda x: x.to_json(), self.method_list))
        result = json.dumps(info)
        return result

    def to_dic(self):
        info = {}
        info["className"] = self.class_name
        info["extends"] = self.extends
        info["id"] = self.id
        info["fields"] = list(map(lambda x: x.to_dic(), self.field_list))
        info["methods"] = list(map(lambda x: x.to_dic(), self.method_list))
        return info