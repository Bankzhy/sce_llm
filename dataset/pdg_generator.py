import copy
import csv
import json
import random
import re

import networkx as nx
import matplotlib.pyplot as plt

from graph.doc_sim import DocSim
from graph.stop_word_remover import StopWordRemover
from graph.cfg_generator import CFGGenerator, CFGNode, CFGEdge
from graph.graph import Graph, Node, Edge
from reflect.sr_statement import SRIFStatement, SRFORStatement, SRWhileStatement, SRStatement, SRTRYStatement
from graph.metrics_calculator import MetricsCalculator

class PDGGenerator:
    def __init__(self, sr_class, sr_method):
        self.sr_class = sr_class
        self.sr_method = sr_method
        self.node_list = []
        self.flow_edge_list = []
        self.id_used = []
        self.max_code_length = 100000
        # var_change : [node_id, node_id...]
        self.var_change_list = {}
        self.dd_edge_list = []
        self.cd_edge_list = []
        self.sp_mark = """#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘'！[\\]^_`{|}~\s]+"""
        self.w2v_model = None
        # self.google_news_path = '/Volumes/rog/research/dataset/model.bin'
        # self.w2v_model = KeyedVectors.load_word2vec_format(self.google_news_path, binary=True)
        # self.kes = KeyWordsExtracotr()
        # self.kes.init_code_corpus()
        self.class_list = []

    def create_graph(self):
        cfg_gen = CFGGenerator(
            sr_method=self.sr_method
        )

        res = cfg_gen.create_graph()
        if res is True:
            print("cfg graph created")
            self.flow_edge_list = cfg_gen.flow_edge_list
            self.node_list = cfg_gen.node_list
            self.id_used = cfg_gen.id_used
            self.__create_data_dependency()
            print("dd created")
            self.__create_control_dependency(cfg_gen)
            print("cd created")
            return True
        else:
            return False

    def __create_control_dependency(self, cfg):
        reverse_graph = self.__reverse_cfg(cfg)

        G = nx.DiGraph()
        start = -1
        for node in reverse_graph.node_list:
            G.add_node(node.id)
            if node.category == 3:
                start = node.id

        for edge in reverse_graph.edge_list:
            G.add_edge(edge.source, edge.target)

        # nx.draw_networkx(G)
        # plt.show()
        i_dominators = sorted(nx.immediate_dominators(G, start).items())
        i_dominators_map = {}
        for i in i_dominators:
            i_dominators_map[i[0]] = i[1]
        # print(i_dominators_map)

        for node in reverse_graph.node_list:
            node.i_dominator = i_dominators_map[node.id]

        dominator_tree = self.__compute_dominators_tree(reverse_graph)
        # self.test(self.node_list, self.flow_edge_list)
        # self.test(dominator_tree.node_list, dominator_tree.edge_list)

        graph = Graph(
            node_list=self.node_list,
            edge_list=self.flow_edge_list
        )

        self.__compute_control_dependency(graph=graph, dom_tree=dominator_tree)

    def __less_common_ancestor(self, tree, a, b):
        ancestors_a = []
        ancestors_b = []
        if tree.get_edge(a, b) is not None:
            return a

        if tree.get_edge(b, a) is not None:
            ancestors_b.append(b)
            return b

        ancestors_a = self.ancestors_of(tree, a)
        ancestors_a.insert(0, a)
        ancestors_b = self.ancestors_of(tree, b)
        ancestors_b.insert(0, b)

        result = self.intersection(ancestors_a, ancestors_b)

        return result[0]

    def __compute_control_dependency(self, graph, dom_tree):
        cdg_node_list = []
        cdg_edge_list = []
        s = self.edges_not_ancestrals_in_tree(graph=graph, tree=dom_tree)
        # print("tree")
        # for eg in dom_tree.edge_list:
        #     print("source: "+str(eg.source)+" , "+"target: "+str(eg.target))
        # print("s")
        # for eg in s:
        #     print("source: "+str(eg.source)+" , "+"target: "+str(eg.target))
        for e in s:
            try:
                a = graph.get_edge_source(e)
                b = graph.get_edge_target(e)
                L = self.__less_common_ancestor(dom_tree, a, b)
                # print("L")
                # print(L.id)
                vert = b
                cdg_node_list.append(a)
                cdg_node_list.append(b)
                edge = PDGCDEdge(
                    id=self.__get_id(),
                    name="",
                    source=a.id,
                    target=b.id
                )
                cdg_edge_list.append(edge)

                while vert is not None:
                    incA = dom_tree.incoming_edges_of(vert)
                    if len(incA) == 0:
                        vert = None
                    else:
                        if len(incA) > 1:
                            raise Exception("More than one incomming edges of the vertex " + str(vert.id))

                        for edge in incA:
                            aux = dom_tree.get_edge_source(edge)
                            if aux.id == L.id:
                                if L.id == a.id:
                                    pass
                                vert = None
                            else:
                                cdg_node_list.append(a)
                                cdg_node_list.append(aux)
                                edge = PDGCDEdge(
                                    id=self.__get_id(),
                                    name="",
                                    source=a.id,
                                    target=aux.id
                                )
                                cdg_edge_list.append(edge)
                                vert = aux
            except Exception as e:
                print(e)

        # self.test(node_list=cdg_node_list, edge_list=cdg_edge_list)
        self.cd_edge_list = cdg_edge_list

    def ancestors_of(self, tree, node):
        ancestors = []
        vert = node
        while vert is not None:
            # print(vert.id)
            incA = tree.incoming_edges_of(vert)
            # print("incA")
            # print(incA[0].id)
            if len(incA) == 0:
                vert = None
            else:
                if len(incA) > 1:
                    raise Exception("More than one incomming edges of the vertex " + str(vert.id))
                for e in incA:
                    ancestors.append(tree.get_edge_source(e))
                    vert = tree.get_edge_source(e)
        return ancestors

    def edges_not_ancestrals_in_tree(self, graph, tree):
        edge_vertex_list = []
        ancestors = []
        for e in graph.edge_list:
            try:
                ancestors = self.ancestors_of(tree, graph.get_edge_source(e))
                ancestors_ids = []
                # print("=======")
                # print("source: " + str(e.source) + " , " + "target: " + str(e.target))
                # print("ances")
                # for n in ancestors:
                #     print("node: " + str(n.id))
                # print("=======")
                for n in ancestors:
                    ancestors_ids.append(n.id)

                if e.target not in ancestors_ids:
                    edge_vertex_list.append(e)
            except Exception as err:
                print(err)
        return edge_vertex_list

    def __get_predecessors(self, node, graph):
        result = []
        edge_set = graph.incoming_edges_of(node)

        for e in edge_set:
            result.append(graph.get_edge_source(e))
        return result

    def intersection(self, new_dom, other_doms):
        result = []
        for v in new_dom:
            if v in other_doms:
                result.append(v)
        return result

    def __compute_dominators(self, graph):
        vertex_list = []
        for node in graph.get_dfs_node_list():
            vertex_list.append(node)

        for v in graph.get_dfs_node_list():
            if v.category != 4:
                v.dominators = copy.copy(vertex_list)
            else:
                l = []
                l.append(v)
                v.dominators = l

        done = False
        while done is False:
            done = True
            for v in graph.get_dfs_node_list():
                length = len(v.dominators)
                if v.category != 4:
                    new_dom = v.dominators
                    pre = self.__get_predecessors(v, graph)
                    # print('pre')
                    # print(pre)
                    for p in pre:
                        new_dom = self.intersection(new_dom, p.dominators)
                    # print('new_dom')
                    # print(new_dom)
                    v.dominators = new_dom
                    v.add_dominators(v)
                if length != len(v.dominators):
                    done = False

    def __compute_i_dominator(self, graph):
        vertex_list = []
        for node in graph.get_dfs_node_list():
            my_doms = node.get_s_dominators()
            i = 0
            while i < len(my_doms):
                dom = my_doms[i]
                intersect_list = self.intersection(my_doms, dom.get_s_dominators())
                my_doms = self.__remove_all_from_list(my_doms, intersect_list)
                i += 1
            # print('md')
            # print(my_doms)
            if len(my_doms) > 1:
                print("Bad")
                raise Exception("there more than one idominators")
            else:
                if len(my_doms) == 1:
                    node.i_dominator = my_doms[0]

    def __remove_all_from_list(self, list1, list2):
        result = []
        for l in list1:
            if l not in list2:
                result.append(l)
        return result

    def __compute_dominators_tree(self, graph):
        dominator_tree_node_list = []
        dominator_tree_edge_list = []

        for node in graph.node_list:
            if node.i_dominator is not None:
                dominator_tree_node_list.append(node)
                dominator_tree_node_list.append(graph.get_node(node.i_dominator))
                if node.i_dominator != node.id:
                    new_edge = Edge(
                        id=self.__get_id(),
                        source=node.i_dominator,
                        target=node.id
                    )
                    dominator_tree_edge_list.append(new_edge)
                # for do in node.i_dominators:
                #     dominator_tree_node_list.append(do)
                #     new_edge = Edge(
                #         id=self.__get_id(),
                #         source=do.id,
                #         target=node.id
                #     )
                #     dominator_tree_edge_list.append(new_edge)

        graph = Graph(
            node_list=dominator_tree_node_list,
            edge_list=dominator_tree_edge_list
        )
        return graph

    def __reverse_cfg(self, cfg):
        node_list = []
        edge_list = []

        for node in cfg.node_list:
            if node.category == cfg.init_index:
                end_fake_st = SRStatement(
                    id=node.sr_statement.id,
                    word_list=['E'],
                    type="Fake"
                )
                new_node = CFGNode(
                    id=node.id,
                    index=node.index,
                    category=cfg.END_ST_C,
                    sr_statement=end_fake_st
                )
                node_list.append(new_node)
            elif node.category == cfg.final_index:
                start_fake_st = SRStatement(
                    id=node.sr_statement.id,
                    word_list=['S'],
                    type="Fake"
                )
                new_node = CFGNode(
                    id=node.id,
                    index=node.index,
                    category=cfg.START_ST_C,
                    sr_statement=start_fake_st
                )
                node_list.append(new_node)
            else:
                new_node = CFGNode(
                    id=node.id,
                    index=node.index,
                    category=node.category,
                    sr_statement=node.sr_statement
                )
                node_list.append(new_node)

        for edge in cfg.flow_edge_list:
            new_edge = CFGEdge(
                id=edge.id,
                source=edge.target,
                target=edge.source,
                name=edge.name
            )
            edge_list.append(new_edge)
        reverse_graph = Graph(
            node_list=node_list,
            edge_list=edge_list
        )
        return reverse_graph

    def __get_id(self):
        new_id = random.randint(0, self.max_code_length)
        if new_id in self.id_used:
            new_id = self.__get_id()
        else:
            self.id_used.append(new_id)
            return new_id

    def __create_data_dependency(self):
        self.__get_var_change_from_statement()
        self.__get_var_use_dependency()
        # print(self.var_change_list)
        # print(self.dd_edge_list)

    def __get_var_change_from_statement(self):
        for node in self.node_list:
            if type(node.sr_statement) != SRIFStatement \
                    and type(node.sr_statement) != SRFORStatement \
                    and type(node.sr_statement) != SRWhileStatement:
                if "=" in node.sr_statement.word_list:
                    for index, word in enumerate(node.sr_statement.word_list):
                        if word == "=" \
                                and node.sr_statement.word_list[index - 1] not in self.sp_mark \
                                and node.sr_statement.word_list[index + 1] not in self.sp_mark:
                            var_change = node.sr_statement.word_list[index - 1]
                            if var_change in self.var_change_list.keys():
                                self.var_change_list[var_change].append(node)
                            else:
                                self.var_change_list[var_change] = [node]

    def __get_var_use_dependency(self):
        for node in self.node_list:
            nswl = node.sr_statement.to_node_word_list()
            if "=" in nswl:
                equal_index = nswl.index("=")
            else:
                equal_index = 0

            for i in range(equal_index + 1, len(nswl)):
                if nswl[i] in self.var_change_list.keys():
                    vcnl = sorted(self.var_change_list[nswl[i]], key=lambda x: x.index, reverse=False)
                    var_change_d = self.__get_nearly_dependency(node, vcnl)
                    if var_change_d is not None:
                        new_dd_edge = PDGDDEdge(
                            id=self.__get_id(),
                            source=node.id,
                            target=var_change_d.id,
                            name=""
                        )
                        self.dd_edge_list.append(new_dd_edge)

    def __get_nearly_dependency(self, var_use_node, var_change_list):
        d_index = 0
        for index, var_change in enumerate(var_change_list):
            if var_use_node.index > var_change.index:
                if index - 1 >= 0:
                    d_index = index - 1
        return var_change_list[d_index]

    def __get_word_similarity_score(self, text1, text2, ):
        ds = DocSim(self.w2v_model)
        source_doc = text1
        target_docs = text2
        sim_scores = ds.calculate_similarity(source_doc, target_docs)
        sim_score = 0
        if len(sim_scores) > 0:
            sim_score = sim_scores[0]["score"]
        sim_score = round(sim_score, 2)
        return sim_score


    def __parse_var_name_text(self, var_name):
        var_list = re.sub(r"([A-Z])", r" \1", var_name).split()
        # print(class_list)
        var_name_lower = []
        for cn in var_list:
            var_name_lower.append(cn.lower())
        if len(var_list) < 2 and len(var_list) > 0:
            var_list.append(var_list[0])
        return " ".join(var_name_lower)

    def __parse_statement_text(self, word_list):
        java_keywords = ["boolean", "int", "long", "short", "byte", "float", "double", "char", "class", "interface",
                         "if", "else", "do", "while", "for", "switch", "case", "default", "break", "continue", "return",
                         "try", "catch", "finally", "public", "protected", "private", "final", "void", "static",
                         "strict", "abstract", "transient", "synchronized", "volatile", "native", "package", "import",
                         "throw", "throws", "extends", "implements", "this", "supper", "instanceof", "new", "true",
                         "false", "null", "goto", "const"]
        special_key = "[\n`~!@#$%^&*()+=\\-_|{}':;',\\[\\].<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。， 、？]"
        statement_text_lower = ""
        for word in word_list:
            if word not in java_keywords and word not in special_key:
                statement_text_lower += self.__parse_var_name_text(word)
        return statement_text_lower

    def __parse_param_text(self, param_list):
        param_text = ""
        for param in param_list:
            param_text += self.__parse_var_name_text(param.name)
        return param_text

    def to_json(self):
        info = {}
        info['nodes'] = []
        info['flow_edges'] = []
        info["dd_edges"] = []
        info["cd_edges"] = []
        info["include_edges"] = []

        metrics_calc = MetricsCalculator(
            sr_class=self.sr_class
        )

        doc_sim = DocSim()

        loc = metrics_calc.get_method_loc(self.sr_method)
        cc = metrics_calc.get_method_cc(self.sr_method)
        pc = metrics_calc.get_method_pc(self.sr_method)
        lcom1 = metrics_calc.get_method_LCOM1(self.sr_method)
        lcom2 = metrics_calc.get_method_LCOM2(self.sr_method)
        lcom3 = metrics_calc.get_method_LCOM3(self.sr_method)
        lcom4 = metrics_calc.get_method_LCOM4(self.sr_method)
        tsmc = metrics_calc.get_tsmc(self.sr_method, doc_sim)
        coh = metrics_calc.get_method_COH(self.sr_method)
        noav = metrics_calc.get_method_noav(self.sr_method)
        class_name_list = [o.class_name for o in self.class_list]
        cd = metrics_calc.get_method_CD(self.sr_method, class_name_list)
        clc = metrics_calc.get_method_clc(self.sr_method)

        new_method_node = {
            'id': self.__get_id(),
            'type': "method",
            "metrics": {
                "loc": loc,
                "cc": cc,
                "pc": pc,
                "lcom1": lcom1,
                "lcom2": lcom2,
                "lcom3": lcom3,
                "lcom4": lcom4,
                "tsmc": tsmc,
                "coh": coh,
                "cd": cd,
                "noav": noav,
                "clc": clc

            }
        }
        info['nodes'].append(new_method_node)

        for node in self.node_list:
            new_statement_node = node.to_dic()

            abcl = metrics_calc.get_statement_abcl(node.sr_statement)
            fuc = metrics_calc.get_statement_fuc(node.sr_statement)
            lmuc = metrics_calc.get_statement_lmuc(node.sr_statement)
            vuc = metrics_calc.get_statement_vuc(node.sr_statement)
            param_name_list = []
            if len(self.sr_method.param_list) > 0:
                for p in self.sr_method.param_list:
                    param_name_list.append(p.name)
            puc = metrics_calc.get_statement_puc(node.sr_statement, param_name_list)
            nbd = metrics_calc.get_statement_block_depth(self.sr_method, node.sr_statement)
            wc = metrics_calc.get_statement_wc(node.sr_statement)
            tsmm = metrics_calc.get_tsmm(sr_method=self.sr_method, sr_statement=node.sr_statement, doc_sim=doc_sim)

            start_line = 0
            end_line = 0

            # if type(node.sr_statement) == SRStatement:
            #     if node.sr_statement.type == "Fake":
            #         start_line = 0
            #         end_line = 0
            #     else:
            #         start_line = node.sr_statement.start_line
            #         end_line = node.sr_statement.end_line
            # else:
            #     start_line = node.sr_statement.start_line
            #     end_line = node.sr_statement.end_line

            new_statement_node["metrics"] = {
                "abcl": abcl,
                "fuc": fuc,
                "lmuc":lmuc,
                "vuc": vuc,
                "puc": puc,
                "nbd": nbd,
                "wc": wc,
                "tsmm": tsmm,
                "start_line": start_line,
                "end_line": end_line,
            }
            info['nodes'].append(new_statement_node)

            new_include_edge = {
                "id": self.__get_id(),
                "source":new_method_node["id"],
                "target":new_statement_node["id"],
                "type": "include"
            }
            info["include_edges"].append(new_include_edge)

        for edge in self.flow_edge_list:
            info['flow_edges'].append(edge.to_dic())
        for edge in self.dd_edge_list:
            info["dd_edges"].append(edge.to_dic())

        for edge in self.cd_edge_list:
            info["cd_edges"].append(edge.to_dic())
        return json.dumps(info)

    def to_database(self, db, project_name, group, extract_lines=""):
        cursor = db.cursor()
        graph_json = self.to_json()
        method_path = project_name+"_"+self.sr_class.class_name+"_"+self.sr_method.get_method_identifier()
        if len(method_path) > 250:
            method_path = method_path[0:250]


        query = (r"insert into lm_master (project, content, class_name, method_name, extract_lines, `group`, split, graph, `path`, label, reviewer_id) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
        values = (project_name, self.sr_method.text, self.sr_class.class_name, self.sr_method.method_name, extract_lines, group, "pool", graph_json, method_path, 9, 0)
        cursor.execute(query, values)
        db.commit()
        # method level metrics


    def to_csv(self, sr_class, sr_method, path, cpsm_ids=[]):
        metrics_calc = MetricsCalculator(
            sr_class=sr_class
        )

        # method.csv
        # node_field_order = ["id", 'loc', 'cc', 'lcom1', 'lcom2', 'lcom3', 'lcom4', 'coh']
        node_field_order = ["id", 'loc', 'cc', 'lcom1', 'coh', 'noav']
        with open(path / "method.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, node_field_order)
            writer.writeheader()
            loc = metrics_calc.get_method_loc(sr_method)
            cc = metrics_calc.get_method_cc(sr_method)
            # pc = metrics_calc.get_method_pc(sr_method)
            lcom1 = metrics_calc.get_method_LCOM1(sr_method)
            lcom2 = metrics_calc.get_method_LCOM2(sr_method)
            coh = metrics_calc.get_method_COH(sr_method)
            noav = metrics_calc.get_method_NOAV(sr_method)

            writer.writerow(dict(zip(node_field_order, [sr_method.id, loc, cc, lcom1, coh, noav])))

            # lcom1 = metrics_calc.get_method_LCOM1(sr_method)
            # lcom2 = metrics_calc.get_method_LCOM2(sr_method)
            # lcom3 = metrics_calc.get_method_LCOM3(sr_method)
            # lcom4 = metrics_calc.get_method_LCOM4(sr_method)
            # coh = metrics_calc.get_method_COH(sr_method)
            #
            # writer.writerow(dict(zip(node_field_order, [sr_method.id, loc, cc, pc, lcom1, lcom2, lcom3, lcom4, coh])))

        # node.csv
        # node_field_order = ["id", 'abcl', 'fuc', 'lmuc', 'bd', 'sim_method_txt', 'sim_param_txt', 'from_cps']
        # node_field_order = ["id", 'abcl', 'fuc', 'lmuc', 'puc', 'nmd', 'vuc', 'wc']
        node_field_order = ["id", 'abcl', 'nbd', 'wc', 'fuc', 'lmuc']
        with open(path / "node.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, node_field_order)
            writer.writeheader()
            for node in self.node_list:
                # basic metrics
                abcl = metrics_calc.get_statement_abcl(node.sr_statement)
                fuc = metrics_calc.get_statement_fuc(node.sr_statement)
                lmuc = metrics_calc.get_statement__lmuc((node.sr_statement))



                block_depth = metrics_calc.get_statement_block_depth(sr_method=sr_method, sr_statement=node.sr_statement)
                if block_depth == -1:
                    block_depth = 0
                param_name_list = []
                if len(sr_method.param_list) > 0:
                    for p in sr_method.param_list:
                        param_name_list.append(p.name)
                puc = metrics_calc.get_statement_puc(sr_statement=node.sr_statement, param_name_list=param_name_list)
                vuc = metrics_calc.get_statement_vuc(sr_statement=node.sr_statement)
                wc = metrics_calc.get_statement_wc(sr_statement=node.sr_statement)
                writer.writerow(dict(zip(node_field_order,
                                         [node.id, abcl, block_depth, wc, fuc, lmuc
                                          ])))


                # # text similarity
                # method_txt_sim = self.__get_word_similarity_score(
                #     text1=self.__parse_statement_text(node.sr_statement.word_list),
                #     text2=self.__parse_var_name_text(sr_method.method_name)
                # )
                # param_txt_sim = self.__get_word_similarity_score(
                #     text1=self.__parse_statement_text(node.sr_statement.word_list),
                #     text2=self.__parse_param_text(sr_method.param_list)
                # )
                #
                # # from copy source method
                # from_cps = 0
                # if node.sr_statement.id in cpsm_ids:
                #     from_cps = 1
                #
                # st = str(node.sr_statement.to_string())
                # # writer.writerow(dict(zip(node_field_order, [node.id, abcl, fuc, lmuc, block_depth, method_txt_sim, param_txt_sim, from_cps])))
                # writer.writerow(dict(zip(node_field_order,
                #                          [node.id, abcl, fuc, lmuc, puc, block_depth, vuc,
                #                           wc])))

        # include_edge.csv
        # include_edge_ids = ["id"]
        # include_edge_source = ["source"]
        # include_edge_target = ["target"]
        # include_edge_similarity = ["similarity"]
        # for node in self.node_list:
        #     include_edge_ids.append(self.__get_id())
        #     include_edge_source.append(sr_method.id)
        #     include_edge_target.append(node.id)
        #
        # with open(path / "include_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, include_edge_ids)
        #     writer.writeheader()
        #     writer.writerow(dict(zip(include_edge_ids, include_edge_source)))
        #     writer.writerow(dict(zip(include_edge_ids, include_edge_target)))

        # cf_edge_index.csv
        cf_field_order = ["id"]
        cf_source = ["source"]
        cf_target = ["target"]

        for edge in self.flow_edge_list:
            cf_field_order.append(edge.id)
            cf_source.append(edge.source)
            cf_target.append(edge.target)

        with open(path / "cf_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, cf_field_order)
            writer.writeheader()
            writer.writerow(dict(zip(cf_field_order, cf_source)))
            writer.writerow(dict(zip(cf_field_order, cf_target)))

        # cd_edge_index.csv
        cd_field_order = ["id"]
        cd_source = ["source"]
        cd_target = ["target"]

        for edge in self.cd_edge_list:
            cd_field_order.append(edge.id)
            cd_source.append(edge.source)
            cd_target.append(edge.target)

        with open(path / "cd_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, cd_field_order)
            writer.writeheader()
            writer.writerow(dict(zip(cd_field_order, cd_source)))
            writer.writerow(dict(zip(cd_field_order, cd_target)))

        # dd_edge_index.csv
        dd_field_order = ["id"]
        dd_source = ["source"]
        dd_target = ["target"]

        for edge in self.dd_edge_list:
            dd_field_order.append(edge.id)
            dd_source.append(edge.source)
            dd_target.append(edge.target)

        with open(path / "dd_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, dd_field_order)
            writer.writeheader()
            writer.writerow(dict(zip(dd_field_order, dd_source)))
            writer.writerow(dict(zip(dd_field_order, dd_target)))

    def to_csv_node(self, sr_class, sr_method, path, word2vec, kes, cpsm_ids=[], use_sid=False):
        metrics_calc = MetricsCalculator(
            sr_class=sr_class
        )
        self.w2v_model = word2vec
        # method.csv
        # node_field_order = ["id", 'loc', 'cc', 'lcom1', 'lcom2', 'lcom3', 'lcom4', 'coh']
        node_field_order = ["id", 'loc', 'cc', 'pc', 'lcom1', 'lcom2', 'lcom3', 'lcom4', 'lcom5', 'coh', 'noav']
        with open(path / "method.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, node_field_order)
            writer.writeheader()

            loc = metrics_calc.get_method_loc(sr_method)
            cc = metrics_calc.get_method_cc(sr_method)
            pc = metrics_calc.get_method_pc(sr_method)
            lcom1 = metrics_calc.get_method_LCOM1(sr_method)
            lcom2 = metrics_calc.get_method_LCOM2(sr_method)
            lcom3 = metrics_calc.get_method_LCOM3(sr_method)
            lcom4 = metrics_calc.get_method_LCOM4(sr_method)
            lcom5 = metrics_calc.get_method_LCOM5(sr_method)
            coh = metrics_calc.get_method_COH(sr_method)
            noav = metrics_calc.get_method_NOAV(sr_method)
            # cd = metrics_calc.get_method_CD(sr_method)
            writer.writerow(dict(zip(node_field_order, [sr_method.id, loc, cc, pc, lcom1, lcom2, lcom3, lcom4, lcom5, coh, noav])))


        # node.csv
        # node_field_order = ["id", 'abcl', 'fuc', 'lmuc', 'bd', 'sim_method_txt', 'sim_param_txt', 'from_cps']
        # node_field_order = ["id", 'abcl', 'fuc', 'lmuc', 'puc', 'nmd', 'vuc', 'wc']
        node_field_order = ["id", 'abcl', 'fuc', 'lmuc', 'puc', 'nbd', 'vuc', 'wc', 'tsmn', 'tsmk', 'from_cps']
        with open(path / "node.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, node_field_order)
            writer.writeheader()
            for node in self.node_list:
                param_name_list = []
                if len(sr_method.param_list) > 0:
                    for p in sr_method.param_list:
                        param_name_list.append(p.name)
                # basic metrics
                abcl = metrics_calc.get_statement_abcl(node.sr_statement)
                fuc = metrics_calc.get_statement_fuc(node.sr_statement)
                lmuc = metrics_calc.get_statement__lmuc((node.sr_statement))
                puc = metrics_calc.get_statement_puc(sr_statement=node.sr_statement, param_name_list=param_name_list)
                block_depth = metrics_calc.get_statement_block_depth(sr_method=sr_method, sr_statement=node.sr_statement)
                if block_depth == -1:
                    block_depth = 0
                vuc = metrics_calc.get_statement_vuc(sr_statement=node.sr_statement)
                wc = metrics_calc.get_statement_wc(sr_statement=node.sr_statement)
                # m_all_words = sr_method.get_all_word()
                # m_all_words_str = " ".join(m_all_words)
                # m_kl = kes.get_keywords(test_doc=m_all_words_str)
                # m_kl = list(m_kl.keys())
                # print("m_kl")
                # print(m_kl)
                # tswkw1 = 0
                # tswkw2 = 0
                # tswkw3 = 0
                # tswp = 0
                # if len(m_kl) >= 3:
                #     tswkw1 = self.__get_word_similarity_score(
                #         text1=self.__parse_statement_text(node.sr_statement.word_list),
                #         text2=m_kl[0]
                #     )
                #     tswkw2 = self.__get_word_similarity_score(
                #         text1=self.__parse_statement_text(node.sr_statement.word_list),
                #         text2=m_kl[1]
                #     )
                #     tswkw3 = self.__get_word_similarity_score(
                #         text1=self.__parse_statement_text(node.sr_statement.word_list),
                #         text2=m_kl[2]
                #     )
                # elif len(m_kl) == 2:
                #     tswkw1 = self.__get_word_similarity_score(
                #         text1=self.__parse_statement_text(node.sr_statement.word_list),
                #         text2=m_kl[0]
                #     )
                #     tswkw2 = self.__get_word_similarity_score(
                #         text1=self.__parse_statement_text(node.sr_statement.word_list),
                #         text2=m_kl[1]
                #     )
                # elif len(m_kl) == 1:
                #     tswkw1 = self.__get_word_similarity_score(
                #         text1=self.__parse_statement_text(node.sr_statement.word_list),
                #         text2=m_kl[0]
                #     )
                # tswp = self.__get_word_similarity_score(
                #     text1=self.__parse_statement_text(node.sr_statement.word_list),
                #     text2=self.__parse_param_text(sr_method.param_list)
                # )



                # # text similarity
                # method_txt_sim = self.__get_word_similarity_score(
                #     text1=self.__parse_statement_text(node.sr_statement.word_list),
                #     text2=self.__parse_var_name_text(sr_method.method_name)
                # )
                # param_txt_sim = self.__get_word_similarity_score(
                #     text1=self.__parse_statement_text(node.sr_statement.word_list),
                #     text2=self.__parse_param_text(sr_method.param_list)
                # )
                #
                # from copy source method

                tsmk = 0
                tsmn = 0
                st_wl = self.remove_stop_word(node.sr_statement.word_list)
                st_wl = self.remove_special_char(st_wl)
                st_wl = self.split_token(st_wl)
                st_wl = self.to_lower(st_wl)


                mn_wl = self.camel_case_split(sr_method.method_name)
                mn_wl = self.to_lower(mn_wl)

                mk_wl = self.remove_stop_word(sr_method.get_all_word())
                mk_wl = self.remove_special_char(mk_wl)
                mk_wl = self.split_token(mk_wl)
                mk_wl = self.to_lower(mk_wl)
                mk_str = " ".join(mk_wl)
                m_kl = kes.get_keywords(test_doc=mk_str)
                m_kl = list(m_kl.keys())

                tsmn = self.__get_word_similarity_score(
                    text1=" ".join(st_wl),
                    text2=" ".join(mn_wl)
                )

                if len(m_kl) > 0:
                    tsmk = self.__get_word_similarity_score(
                        text1=" ".join(st_wl),
                        text2=" ".join(m_kl)
                    )



                from_cps = 0
                if use_sid is True:
                    if -1 in cpsm_ids:
                        from_cps = -1
                    sr_method.refresh_sid()
                    if node.sr_statement.sid in cpsm_ids:
                        from_cps = 1
                else:
                    if node.sr_statement.id in cpsm_ids:
                        from_cps = 1

                # st = str(node.sr_statement.to_string())
                # # writer.writerow(dict(zip(node_field_order, [node.id, abcl, fuc, lmuc, block_depth, method_txt_sim, param_txt_sim, from_cps])))
                writer.writerow(dict(zip(node_field_order,
                                         [node.id, abcl, fuc, lmuc, puc, block_depth, vuc,
                                          wc, tsmn, tsmk, from_cps])))

        # include_edge.csv
        # include_edge_ids = ["id"]
        # include_edge_source = ["source"]
        # include_edge_target = ["target"]
        # include_edge_similarity = ["similarity"]
        # for node in self.node_list:
        #     include_edge_ids.append(self.__get_id())
        #     include_edge_source.append(sr_method.id)
        #     include_edge_target.append(node.id)
        #
        # with open(path / "include_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, include_edge_ids)
        #     writer.writeheader()
        #     writer.writerow(dict(zip(include_edge_ids, include_edge_source)))
        #     writer.writerow(dict(zip(include_edge_ids, include_edge_target)))

        # cf_edge_index.csv
        cf_field_order = ["id"]
        cf_source = ["source"]
        cf_target = ["target"]

        for edge in self.flow_edge_list:
            cf_field_order.append(edge.id)
            cf_source.append(edge.source)
            cf_target.append(edge.target)

        with open(path / "cf_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, cf_field_order)
            writer.writeheader()
            writer.writerow(dict(zip(cf_field_order, cf_source)))
            writer.writerow(dict(zip(cf_field_order, cf_target)))

        # cd_edge_index.csv
        cd_field_order = ["id"]
        cd_source = ["source"]
        cd_target = ["target"]

        for edge in self.cd_edge_list:
            cd_field_order.append(edge.id)
            cd_source.append(edge.source)
            cd_target.append(edge.target)

        with open(path / "cd_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, cd_field_order)
            writer.writeheader()
            writer.writerow(dict(zip(cd_field_order, cd_source)))
            writer.writerow(dict(zip(cd_field_order, cd_target)))

        # dd_edge_index.csv
        dd_field_order = ["id"]
        dd_source = ["source"]
        dd_target = ["target"]

        for edge in self.dd_edge_list:
            dd_field_order.append(edge.id)
            dd_source.append(edge.source)
            dd_target.append(edge.target)

        with open(path / "dd_edge_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, dd_field_order)
            writer.writeheader()
            writer.writerow(dict(zip(dd_field_order, dd_source)))
            writer.writerow(dict(zip(dd_field_order, dd_target)))

    def test(self, node_list, edge_list):
        G = nx.DiGraph()  # 有向グラフ (Directed Graph)

        # 頂点の追加
        for node in node_list:
            G.add_node(node.id)

        # 辺の追加 (頂点も必要に応じて追加されます)
        for edge in edge_list:
            G.add_edge(edge.source, edge.target)

        nx.draw_networkx(G)
        plt.show()

    def show_cfg(self):
        G = nx.DiGraph()  # 有向グラフ (Directed Graph)

        # 頂点の追加
        for node in self.node_list:
            G.add_node(node.id)

        # 辺の追加 (頂点も必要に応じて追加されます)
        for edge in self.flow_edge_list:
            G.add_edge(edge.source, edge.target)

        nx.draw_networkx(G)
        plt.show()

    def remove_stop_word(self, word_list):
        result_list = StopWordRemover().remove_stop_word(word_list=word_list)
        return result_list

    def remove_special_char(self, word_list):
        result_list = []
        pattern = "[^A-Za-z]+"
        for word in word_list:
            match_obj = re.match(pattern, word, re.M | re.I)
            if match_obj is None:
                result_list.append(word)
        return result_list


    def to_lower(self, word_list):
        result = []
        for word in word_list:
            result.append(word.lower())
        return result

    def split_token(self, word_list):
        result = []
        for word in word_list:
            ws = self.camel_case_split(word)
            result.extend(ws)
        result = self.under_case_split(result)
        return result

    def under_case_split(self, word_list):
        result = []
        for w in word_list:
            ws = w.split("_")
            result.extend(ws)
        return result

    def camel_case_split(self, str):
        if len(str) == 0:
            return ""
        words = [[str[0]]]

        for c in str[1:]:
            if words[-1][-1].islower() and c.isupper():
                words.append(list(c))
            else:
                words[-1].append(c)

        return [''.join(word) for word in words]


class PDGDDEdge:
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


class PDGCDEdge:
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
