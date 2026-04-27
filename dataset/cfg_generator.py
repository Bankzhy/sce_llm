import json
import random
import uuid

from reflect.sr_statement import SRIFStatement, SRFORStatement, SRWhileStatement, SRTRYStatement, SRStatement, \
    SRSwitchStatement


class CFGGenerator:
    def __init__(self, sr_method):
        self.sr_method = sr_method
        self.node_list = []
        self.flow_edge_list = []
        self.id_used = []
        self.break_st_list = []
        self.max_code_length = 100000000
        self.final_index = -10
        self.init_index = 0
        self.end_block = -1
        self.current_index = 1

        self.NORMAL_ST_C = 0
        self.IF_ST_C = 1
        self.FOR_ST_C = 2
        self.END_ST_C = 3
        self.START_ST_C = 4
        self.TRY_ST_C = 5
        self.SWITCH_ST_C = 6

        self.start_node = None
        self.end_node = None

    def create_graph(self):

        # self.flow_edge_list, self.node_list = self.get_node_flow_list(self.sr_method.statement_list)
        # self.node_list, self.flow_edge_list, block_final_index = self.__parse_block_statement(
        #     statement_list=self.sr_method.statement_list,
        #     start_index=0,
        #     depth=[0]
        # )
        try:
            start_fake_st = SRStatement(
                id=uuid.uuid1().hex,
                word_list=['S'],
                type="Fake"
            )
            end_fake_st = SRStatement(
                id=uuid.uuid1().hex,
                word_list=['E'],
                type="Fake"
            )

            self.start_node = self.__create_new_node(st=start_fake_st, index=self.init_index)
            self.end_node = self.__create_new_node(st=end_fake_st, index=self.final_index)

            self.node_list, self.flow_edge_list = self.__parse_block_to_graph(self.sr_method.statement_list)

            start_edge = self.__create_new_edge(source=self.start_node.id, target=self.node_list[0].id, name="")
            self.flow_edge_list = self.__exchange_block_final(new_id=self.end_node.id, edge_list=self.flow_edge_list)

            self.node_list.append(self.start_node)
            self.node_list.append(self.end_node)
            self.flow_edge_list.append(start_edge)
            return True
        except Exception as e:
            print(e)
            return False

    def __exchange_block_final(self, new_id, edge_list):
        for edge in edge_list:
            if edge.target == -1:
                edge.target = new_id
            elif edge.target == -10:
                edge.target = new_id
        return edge_list

    def __exchange_edge_target(self, source, new_target, edge_list):
        for i in range(0, len(edge_list)):
            if edge_list[i].source == source:
                edge_list[i].target = new_target
        return edge_list

    def __delete_edge(self, source, target, edge_list):
        for i in range(0, len(edge_list)):
            if edge_list[i].source == source and edge_list[i].target == target:
                del edge_list[i]
                break
        return edge_list

    def __parse_block_to_graph(self, statement_list):
        node_list = []
        edge_list = []
        ex_node_list = []
        ex_edge_list = []

        # create node
        for index, statement in enumerate(statement_list):
            new_node = self.__create_new_node(st=statement, index=self.current_index)
            node_list.append(new_node)
            self.current_index += 1

        # create edge
        for index, node in enumerate(node_list):
            if index < len(node_list)-1:
                new_edge = self.__create_new_edge(source=node.id, target=node_list[index+1].id, name="")
                edge_list.append(new_edge)
            else:
                new_edge = self.__create_new_edge(source=node.id, target=self.end_block, name="")
                edge_list.append(new_edge)

        for index, node in enumerate(node_list):
            if node.category == self.IF_ST_C:
                # pos block
                pos_node_list, pos_edge_list = self.__parse_block_to_graph(node.sr_statement.pos_statement_list)
                if len(pos_node_list) == 0:
                    continue
                dominator_pos_edge = self.__create_new_edge(source=node.id, target=pos_node_list[0].id, name="T")
                if index < len(node_list)-1:
                    pos_edge_list = self.__exchange_block_final(node_list[index + 1].id, pos_edge_list)
                ex_node_list.extend(pos_node_list)
                ex_edge_list.extend(pos_edge_list)
                ex_edge_list.append(dominator_pos_edge)

                # neg block
                if len(node.sr_statement.neg_statement_list) > 0:
                    neg_node_list, neg_edge_list = self.__parse_block_to_graph(node.sr_statement.neg_statement_list)
                    dominator_neg_edge = self.__create_new_edge(source=node.id, target=neg_node_list[0].id, name="F")
                    if index < len(node_list) - 1:
                        edge_list = self.__delete_edge(source=node.id, target=node_list[index + 1].id, edge_list=edge_list)
                        neg_edge_list = self.__exchange_block_final(node_list[index + 1].id, neg_edge_list)
                    else:
                        edge_list = self.__delete_edge(source=node.id, target=self.end_block, edge_list=edge_list)

                    ex_node_list.extend(neg_node_list)
                    ex_edge_list.extend(neg_edge_list)
                    ex_edge_list.append(dominator_neg_edge)
            elif node.category == self.SWITCH_ST_C:
                if len(node.sr_statement.switch_case_list) > 0:
                    for sc in node.sr_statement.switch_case_list:
                        sc_node = self.__create_new_node(st=sc, index=self.current_index)
                        sc_edge = self.__create_new_edge(source=node.id, target=sc_node.id, name="")
                        sc_child_node_list, sc_child_edge_list = self.__parse_block_to_graph(sc.statement_list)

                        if index < len(node_list) - 1:
                            sc_child_edge_list = self.__exchange_block_final(node_list[index + 1].id, sc_child_edge_list)


                        domi_edge = self.__create_new_edge(source=sc_node.id, target=sc_child_node_list[0].id, name="")
                        ex_node_list.append(sc_node)
                        ex_node_list.extend(sc_child_node_list)
                        ex_edge_list.append(sc_edge)
                        ex_edge_list.extend(sc_child_edge_list)
                        ex_edge_list.append(domi_edge )

            elif node.category == self.FOR_ST_C:
                child_node_list, child_edge_list = self.__parse_block_to_graph(node.sr_statement.child_statement_list)
                dominator_child_edge = self.__create_new_edge(source=node.id, target=child_node_list[0].id, name="")
                child_edge_list = self.__exchange_block_final(node.id, child_edge_list)
                for cn in child_node_list:
                    if len(cn.sr_statement.word_list) == 0:
                        continue
                    if cn.sr_statement.word_list[0] == "break" and cn.id not in self.break_st_list:
                        child_edge_list = self.__exchange_edge_target(source=cn.id, new_target=self.end_block,
                                                                   edge_list=child_edge_list)
                        self.break_st_list.append(cn.id)


                ex_node_list.extend(child_node_list)
                ex_edge_list.extend(child_edge_list)
                ex_edge_list.append(dominator_child_edge)

            elif node.category == self.TRY_ST_C:
                try_node_list, try_edge_list = self.__parse_block_to_graph(node.sr_statement.try_statement_list)
                dominator_try_edge = self.__create_new_edge(source=node.id, target=try_node_list[0].id, name="DF")

                catch_node_list = []
                catch_edge_list = []
                if len(node.sr_statement.catch_block_list) > 0:
                    cb_st_list =[]
                    for cb in node.sr_statement.catch_block_list:
                        cb_st_list.append(cb.to_if_st_expression())
                    catch_node_list, catch_edge_list = self.__parse_block_to_graph(cb_st_list)

                final_node_list = []
                final_edge_list = []
                if len(node.sr_statement.final_block_statement_list) > 0:
                    final_node_list, final_edge_list = self.__parse_block_to_graph(node.sr_statement.final_block_statement_list)
                    if len(final_node_list) > 0:
                        try_edge_list = self.__exchange_block_final(new_id=final_node_list[0].id,
                                                                    edge_list=try_edge_list)
                        catch_edge_list = self.__exchange_block_final(new_id=final_node_list[0].id,
                                                                    edge_list=catch_edge_list)


                ex_node_list.extend(try_node_list)
                ex_edge_list.extend(try_edge_list)
                ex_edge_list.append(dominator_try_edge)

                if len(catch_node_list) > 0:
                    do_catch_edge = self.__create_new_edge(source=node.id, target=catch_node_list[0].id, name="")

                    ex_edge_list.append(do_catch_edge)
                    ex_node_list.extend(catch_node_list)
                    ex_edge_list.extend(catch_edge_list)

                if len(final_node_list) > 0:
                    do_final_edge = self.__create_new_edge(source=node.id, target=final_node_list[0].id, name="")

                    ex_edge_list.append(do_final_edge)
                    ex_node_list.extend(final_node_list)
                    ex_edge_list.extend(final_edge_list)

        node_list.extend(ex_node_list)
        edge_list.extend(ex_edge_list)

        for cn in node_list:
            if len(cn.sr_statement.word_list) > 0:
                if cn.sr_statement.word_list[0] == "return":
                    edge_list = self.__exchange_edge_target(source=cn.id, new_target=self.end_node.id,
                                                                  edge_list=edge_list)

        return node_list, edge_list


    # def __parse_block_statement(self, statement_list, start_index, depth=[0]):
    #     node_list = []
    #     edge_list = []
    #     current_index = start_index
    #
    #     for index, statement in enumerate(statement_list):
    #         current_index += 1
    #         new_node = self.__create_new_node(st=statement, index=current_index)
    #         if index == 0:
    #             new_edge = self.__create_new_edge(source=depth[len(depth)-1], target=current_index, name="")
    #         else:
    #             new_edge = self.__create_new_edge(source=current_index-1, target=current_index, name="")
    #         node_list.append(new_node)
    #         edge_list.append(new_edge)
    #
    #         if len(depth) == 1 and index == len(statement_list)-1:
    #
    #             end_edge = self.__create_new_edge(source=current_index, target=self.final_index, name="")
    #             edge_list.append(end_edge)
    #
    #
    #         if type(statement) == SRIFStatement:
    #             new_depth = []
    #             new_depth.extend(depth)
    #             new_depth.append(current_index)
    #             pos_node_list, pos_edge_list, pos_final_index = self.__parse_block_statement(
    #                 statement_list=statement.pos_statement_list,
    #                 start_index=current_index,
    #                 depth=new_depth
    #             )
    #             node_list.extend(pos_node_list)
    #             edge_list.extend(pos_edge_list)
    #
    #             new_depth = []
    #             new_depth.extend(depth)
    #             new_depth.append(current_index)
    #             neg_node_list, neg_edge_list, neg_final_index = self.__parse_block_statement(
    #                 statement_list=statement.neg_statement_list,
    #                 start_index=pos_final_index,
    #                 depth=new_depth
    #             )
    #             node_list.extend(neg_node_list)
    #             edge_list.extend(neg_edge_list)
    #
    #             current_index = neg_final_index
    #             if index < len(statement_list)-1:
    #                 new_edge = self.__create_new_edge(source=pos_final_index, target=current_index+1, name="")
    #                 edge_list.append(new_edge)
    #
    #         elif type(statement) == SRFORStatement:
    #             new_depth = []
    #             new_depth.extend(depth)
    #             new_depth.append(current_index)
    #             child_node_list, child_edge_list, child_final_index = self.__parse_block_statement(
    #                 statement_list=statement.child_statement_list,
    #                 start_index=current_index,
    #                 depth=new_depth
    #             )
    #
    #             node_list.extend(child_node_list)
    #             edge_list.extend(child_edge_list)
    #             new_edge = self.__create_new_edge(source=child_final_index, target=current_index, name="")
    #             edge_list.append(new_edge)
    #             current_index = child_final_index
    #
    #     return node_list, edge_list, current_index


    def __get_id(self):
        new_id = random.randint(0, self.max_code_length)
        if new_id in self.id_used:
            new_id = self.__get_id()
            return new_id
        else:
            self.id_used.append(new_id)
            return new_id

    def __create_new_node(self, st, index):
        ctg = self.NORMAL_ST_C
        if type(st) == SRIFStatement:
            ctg = self.IF_ST_C
        elif type(st) == SRFORStatement or type(st) == SRWhileStatement:
            ctg = self.FOR_ST_C
        elif type(st) == SRTRYStatement:
            ctg = self.TRY_ST_C
        elif type(st) == SRSwitchStatement:
            ctg = self.SWITCH_ST_C

        if index == self.init_index:
            ctg = self.START_ST_C
        elif index == self.final_index:
            ctg = self.END_ST_C

        # node = {
        #     "sid": st.id,
        #     "name": st.pindex,
        #     "id": str(st.pindex),
        #     "category": ctg,
        #     "value": st.get_statement_string()
        # }

        node = CFGNode(
            id=self.__get_id(),
            index=index,
            sr_statement=st,
            category=ctg
        )
        return node

    def __create_new_edge(self, source, target, name):
        edge = CFGEdge(
            id=self.__get_id(),
            source=source,
            target=target,
            name=name
        )
        return edge

    # def get_node_flow_list(self, statement_list, start_index=0):
    #     flow_list = []
    #     node_list = []
    #     current_index = start_index
    #
    #     for index, st in enumerate(statement_list):
    #
    #         if type(st) == SRIFStatement:
    #             dominate_index = current_index
    #             pos_end_index = current_index
    #             neg_end_index = current_index
    #             pos_return = False
    #             neg_return = False
    #             # st.set_pindex(current_index)
    #             node_list.append(self.__create_new_node(st, current_index))
    #
    #             if len(st.pos_statement_list) > 0:
    #                 dominate_flow = CFGFlowEdge(
    #                     source=dominate_index,
    #                     target=current_index+1,
    #                     name="True",
    #                     id=self.__get_id()
    #                 )
    #                 flow_list.append(dominate_flow)
    #                 pos_block_flow_list, pos_block_node_list = self.get_node_flow_list(
    #                     statement_list=st.pos_statement_list,
    #                     start_index=dominate_index+1
    #                 )
    #                 node_list.extend(pos_block_node_list)
    #
    #                 if pos_block_flow_list[len(pos_block_flow_list)-1].target == -1 \
    #                         or pos_block_flow_list[len(pos_block_flow_list)-1].target == -2:
    #                     pos_block_flow_list.pop()
    #                     # current_index-=1
    #                 if pos_block_node_list[len(pos_block_node_list)-1].sr_statement.word_list[0]=="return":
    #                     pos_return = True
    #
    #                 flow_list.extend(pos_block_flow_list)
    #                 current_index = current_index + len(pos_block_node_list)
    #                 pos_end_index = current_index
    #
    #             if len(st.neg_statement_list) > 0:
    #                 dominate_flow = CFGFlowEdge(
    #                     source=dominate_index,
    #                     target=current_index + 1,
    #                     name="False",
    #                     id=self.__get_id()
    #                 )
    #                 flow_list.append(dominate_flow)
    #                 neg_block_flow_list, neg_block_node_list = self.get_node_flow_list(
    #                     statement_list=st.neg_statement_list,
    #                     start_index=current_index+1
    #                 )
    #                 node_list.extend(neg_block_node_list)
    #
    #                 if neg_block_flow_list[len(neg_block_flow_list)-1].target == -1 or neg_block_flow_list[len(neg_block_flow_list)-1].target == -2:
    #                     neg_block_flow_list.pop()
    #
    #                 if neg_block_node_list[len(neg_block_node_list)-1].sr_statement.word_list[0] == "return":
    #                     pos_return = True
    #
    #                 flow_list.extend(neg_block_flow_list)
    #                 current_index = current_index + len(neg_block_node_list)
    #                 neg_end_index = current_index
    #             else:
    #                 neg_end_index = dominate_index
    #
    #             # final connection
    #             if index == (len(statement_list)-1):
    #                 if pos_return is True:
    #                     pos_final_flow = CFGFlowEdge(
    #                         source=pos_end_index,
    #                         target=-2,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 else:
    #                     pos_final_flow = CFGFlowEdge(
    #                         source=pos_end_index,
    #                         target=-1,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #
    #                 if neg_return is True:
    #                     neg_final_flow = CFGFlowEdge(
    #                         source=neg_end_index,
    #                         target=-2,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 else:
    #                     neg_final_flow = CFGFlowEdge(
    #                         source=neg_end_index,
    #                         target=-1,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 flow_list.append(pos_final_flow)
    #                 flow_list.append(neg_final_flow)
    #             else:
    #                 if pos_return is True:
    #                     pos_final_flow = CFGFlowEdge(
    #                         source=pos_end_index,
    #                         target=-2,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 else:
    #                     pos_final_flow = CFGFlowEdge(
    #                         source=pos_end_index,
    #                         target=current_index+1,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #
    #                 if neg_return is True:
    #                     neg_final_flow = CFGFlowEdge(
    #                         source=neg_end_index,
    #                         target=-2,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 else:
    #                     neg_final_flow = CFGFlowEdge(
    #                         source=neg_end_index,
    #                         target=current_index+1,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 current_index += 1
    #                 flow_list.append(pos_final_flow)
    #                 flow_list.append(neg_final_flow)
    #         elif type(st) == SRFORStatement or type(st) == SRWhileStatement:
    #             dominate_index = current_index
    #             # st.set_pindex(current_index)
    #             node_list.append(self.__create_new_node(st, current_index))
    #
    #             dominate_flow = CFGFlowEdge(
    #                 source=dominate_index,
    #                 target=current_index + 1,
    #                 name="",
    #                 id=self.__get_id()
    #             )
    #             flow_list.append(dominate_flow)
    #             child_block_flow_list, child_block_node_list = self.get_node_flow_list(
    #                 statement_list=st.child_statement_list,
    #                 start_index=dominate_index+1
    #             )
    #             node_list.extend(child_block_node_list)
    #
    #             if child_block_flow_list[len(child_block_flow_list)-1].target == -1:
    #                 child_block_flow_list[len(child_block_flow_list) - 1].target = dominate_index
    #             elif child_block_flow_list[len(child_block_flow_list)-1].target == -2:
    #                 child_block_flow_list[len(child_block_flow_list) - 1].target = -1
    #
    #             if child_block_flow_list[len(child_block_flow_list)-2].target == -1:
    #                 child_block_flow_list[len(child_block_flow_list) - 2].target = dominate_index
    #             elif child_block_flow_list[len(child_block_flow_list)-2].target == -2:
    #                 child_block_flow_list[len(child_block_flow_list) - 2].target = -1
    #
    #             flow_list.extend(child_block_flow_list)
    #             current_index = current_index + len(child_block_node_list)
    #
    #             #final flow
    #             if index == (len(statement_list) - 1):
    #                 next_flow = CFGFlowEdge(
    #                     source=dominate_index,
    #                     target=-1,
    #                     name="",
    #                     id=self.__get_id()
    #                 )
    #                 flow_list.append(next_flow)
    #             else:
    #                 next_flow = CFGFlowEdge(
    #                     source=dominate_index,
    #                     target=current_index+1,
    #                     name="",
    #                     id=self.__get_id()
    #                 )
    #                 current_index +=1
    #                 flow_list.append(next_flow)
    #         elif type(st) == SRTRYStatement:
    #             dominate_index = current_index
    #             try_end_index = current_index
    #             catch_end_index = current_index
    #             # st.set_pindex(current_index)
    #             node_list.append(self.__create_new_node(st, current_index))
    #
    #             if len(st.try_statement_list) > 0:
    #                 dominate_flow = CFGFlowEdge(
    #                     source=dominate_index,
    #                     target=current_index + 1,
    #                     name="try",
    #                     id=self.__get_id()
    #                 )
    #                 flow_list.append(dominate_flow)
    #                 try_block_flow_list, try_block_node_list = self.get_node_flow_list(
    #                     statement_list=st.try_statement_list,
    #                     start_index=dominate_index + 1
    #                 )
    #                 node_list.extend(try_block_node_list)
    #
    #                 if try_block_flow_list[len(try_block_flow_list) - 1].target == -1:
    #                     try_block_flow_list.pop()
    #                     # current_index-=1
    #                 flow_list.extend(try_block_flow_list)
    #                 current_index = current_index + len(try_block_node_list)
    #                 try_end_index = current_index
    #
    #             if len(st.catch_statement_list) > 0:
    #                 dominate_flow = CFGFlowEdge(
    #                     source=dominate_index,
    #                     target=current_index + 1,
    #                     name="catch",
    #                     id=self.__get_id()
    #                 )
    #                 flow_list.append(dominate_flow)
    #                 catch_block_flow_list, catch_block_node_list = self.get_node_flow_list(
    #                     statement_list=st.catch_statement_list,
    #                     start_index=current_index + 1
    #                 )
    #                 node_list.extend(catch_block_node_list)
    #
    #                 # if catch_block_flow_list[len(catch_block_flow_list) - 1].target == -1:
    #                 #     catch_block_flow_list.pop()
    #
    #                 flow_list.extend(catch_block_flow_list)
    #                 current_index = current_index + len(catch_block_node_list)
    #                 catch_end_index = current_index
    #             else:
    #                 catch_end_index = dominate_index
    #
    #             # final connection
    #             if index == (len(statement_list) - 1):
    #                 try_final_flow = CFGFlowEdge(
    #                     source=try_end_index,
    #                     target=-1,
    #                     name="",
    #                     id=self.__get_id()
    #                 )
    #                 # neg_final_flow = Flow(
    #                 #     source=neg_end_index,
    #                 #     target=-1,
    #                 #     name=""
    #                 # )
    #                 flow_list.append(try_final_flow)
    #                 # flow_list.append(neg_final_flow)
    #             else:
    #                 try_final_flow = CFGFlowEdge(
    #                     source=try_end_index,
    #                     target=current_index + 1,
    #                     name="",
    #                     id=self.__get_id()
    #                 )
    #                 # neg_final_flow = Flow(
    #                 #     source=neg_end_index,
    #                 #     target=current_index + 1,
    #                 #     name=""
    #                 # )
    #                 current_index += 1
    #                 flow_list.append(try_final_flow)
    #                 # flow_list.append(neg_final_flow)
    #         else:
    #             if (index+1) != len(statement_list) and st.word_list[0] != "return":
    #                 # st.set_pindex(current_index)
    #                 node_list.append(self.__create_new_node(st, current_index))
    #
    #                 new_flow = CFGFlowEdge(
    #                     source=current_index,
    #                     target=current_index+1,
    #                     name="",
    #                     id=self.__get_id()
    #                 )
    #                 current_index += 1
    #                 flow_list.append(new_flow)
    #             else:
    #
    #
    #                 # st.set_pindex(current_index)
    #                 node_list.append(self.__create_new_node(st, current_index))
    #
    #                 if st.word_list[0] == "return":
    #                     end_flow = CFGFlowEdge(
    #                         source=current_index,
    #                         target=-2,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 else:
    #                     end_flow = CFGFlowEdge(
    #                         source=current_index,
    #                         target=-1,
    #                         name="",
    #                         id=self.__get_id()
    #                     )
    #                 current_index += 1
    #                 flow_list.append(end_flow)
    #     return flow_list, node_list

    def to_json(self):
        info = {}
        info['nodes'] = []
        info['flowEdges'] = []
        for node in self.node_list:
            info['nodes'].append(node.to_dic())
        for edge in self.flow_edge_list:
            info['flowEdges'].append(edge.to_dic())

        return json.dumps(info)

    def to_diGraph(self):
        result = "diGraph " + self.sr_method.method_name + " {" + "\n"
        node_map = {}
        for index, node in enumerate(self.node_list):
            shape="rectangle"
            if node.sr_statement.type == "return_statement":
                shape = "parallelogram"
            elif node.sr_statement.type == "if_statement":
                shape = "diamond"
            elif node.sr_statement.type == "for_statement":
                shape = "hexagon"
            if node.sr_statement.type != "Fake":
                node_map[node.id] = index + 1
                result += f"\t{index+1}[label={node.sr_statement.to_node_string()}, shape={shape}]"
                result += "\n"
        for edge in self.flow_edge_list:
            if edge.source in node_map.keys() and edge.target in node_map.keys():
                result += f'\t{node_map[edge.source]}->{node_map[edge.target]}'
                result += "\n"
        result += "}"
        return result

class CFGNode:
    def __init__(self, id, index, category, sr_statement):
        self.id = id
        self.index = index
        self.category = category
        self.sr_statement = sr_statement
        self.dominators = []
        self.i_dominator = None

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['index'] = self.index
        info['category'] = self.category
        info['value'] = self.sr_statement.to_node_string()
        return info

    def add_dominators(self, node):
        if node not in self.dominators:
            self.dominators.append(node)

    def get_s_dominators(self):
        result = []
        for n in self.dominators:
            if n.id != self.id:
                result.append(n)
        return result


class CFGEdge:
    def __init__(self, id, source, target, name):
        self.id = id
        self.source = source
        self.target = target
        self.name = name

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['name'] = self.name
        return info


class CFGFlowEdge(CFGEdge):
    def __init__(self, id, source, target, name):
        self.id = id
        self.source = source
        self.target = target
        self.name = name
