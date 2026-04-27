import json

from reflect.sr_core import SRCore


class SRProgram(SRCore):
    def __init__(self, program_name, program_id, class_list=[], package_name="", import_list=[]):
        self.program_name = program_name
        self.program_id = program_id
        self.class_list = class_list
        self.import_list = import_list
        self.package_name = package_name

    def to_string(self):
        result = ""
        result += self.package_name
        result += "\n"

        for ip in self.import_list:
            result += ip
            result += "\n"

        for cl in self.class_list:
            result += cl.to_string()
            result += "\n"
        return result

    def to_json(self):
        info={}
        info["programName"] = self.program_name
        info["programId"] = self.program_id
        info["classList"] = list(map(lambda x: x.to_json(), self.class_list))
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info["programName"] = str(self.program_name)
        info["programId"] = self.program_id
        info["classList"] = list(map(lambda x: x.to_dic(), self.class_list))
        return info