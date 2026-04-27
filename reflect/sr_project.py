import json


class SRProject:
    def __init__(self, project_name, project_id, program_list=[]):
        self.project_name = project_name
        self.project_id = project_id
        self.program_list = program_list

    def to_json(self):
        info = {}
        info["projectId"] = self.project_id
        info["projectName"] = str(self.project_name)
        info["programList"] = list(map(lambda x: x.to_dic(), self.program_list))
        return json.dumps(info)