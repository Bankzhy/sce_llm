import json

from reflect.sr_core import SRCore


class SRField(SRCore):
    def __init__(self, id, word_list=[], field_type="", field_name="", field_value=None, modifiers=""):
        self.word_list = word_list
        self.field_type = field_type
        self.field_name = field_name
        self.field_value = field_value
        self.id = id
        self.modifiers = modifiers

    def to_string(self):
        return " ".join(self.word_list)

    def to_json(self):
        field_info = {}
        field_info['fieldName'] = self.field_name
        field_info['fieldType'] = self.field_type
        field_info['fieldValue'] = self.field_value
        field_info['modifiers'] = self.modifiers
        field_info['id'] = self.id
        return json.dumps(field_info)

    def to_dic(self):
        field_info = {}
        field_info['fieldName'] = self.field_name
        field_info['fieldType'] = self.field_type
        field_info['fieldValue'] = self.field_value
        field_info['modifiers'] = self.modifiers
        field_info['id'] = self.id
        return field_info
