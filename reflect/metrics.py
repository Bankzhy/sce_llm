import math

from reflect.sr_statement import SRIFStatement, SRFORStatement, SRWhileStatement, SRTRYStatement, SRSwitchStatement

import numpy as np

from reflect.tf_idf import TFIDF


class ClassLevelMetrics:
    def __init__(self, class_list):
        self.class_list = class_list

    def get_LOC(self, cls):
        loc = cls.end_line - cls.start_line + 1
        return loc

    # Number of Methods
    def get_NOM(self, cls):
        method_count = 0

        method_count += len(cls.method_list)
        return method_count

    # The metric is a count of the number of public methods in a class
    def get_CIS(self, cls):
        p_method_count = 0

        for method in cls.method_list:
            if len(method.modifiers) > 0:
                if method.modifiers[0] == "public":
                    p_method_count += 1
        return p_method_count

    # Number of Attributes
    def get_NOA(self, cls):
        field_count = 0

        field_count += len(cls.field_list)
        return field_count

    # Number of Public Attributes
    def get_NOPA(self, cls):
        p_field_count = 0

        for f in cls.field_list:
            if len(f.modifiers) > 0:
                if "public" in f.modifiers:
                    p_field_count += 1
        return p_field_count

    def get_LCOM(self, cls):
        result = 0

        i_l = []
        P = 0
        Q = 0
        num_methods = len(cls.method_list)
        field_list = cls.field_list
        for f in field_list:
            # if "static" not in f.modifiers:
                i_l.append(f)

        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                intersection_il = self.get_intersection_il(method1=cls.method_list[i], method2=cls.method_list[j], i_l=i_l)
                if len(intersection_il) > 0:
                    Q += 1
                else:
                    P += 1
        if P > Q:
            result = P - Q
        else:
            result = 0
        return result

    def get_LCOM_p(self, cls, method_list):
        result = 0

        i_l = []
        P = 0
        Q = 0
        num_methods = len(method_list)
        field_list = cls.field_list
        for f in field_list:
            # if "static" not in f.modifiers:
            i_l.append(f)

        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                intersection_il = self.get_intersection_il(method1=method_list[i], method2=method_list[j],
                                                           i_l=i_l)
                if len(intersection_il) > 0:
                    Q += 1
                else:
                    P += 1
        if P > Q:
            result = P - Q
        else:
            result = 0
        return result

    def get_C3(self, method_list, w2v=None):
        num_method = len(method_list)

        if num_method == 1:
            return 0

        matrix = np.zeros((num_method, num_method))

        tf_idf = TFIDF()
        tf_idf.calc_with_statements(method_list)
        processed_method_list = tf_idf.processed_method_list
        tf_idf_vectors = tf_idf.tfIdf_vectors
        euclidean_norm = tf_idf.euclidean_norm

        # print(self.num_method)
        for i in range(0, num_method):
            for j in range(i, num_method):
                if i == j:
                    matrix[i][j] += 1.0
                    continue

                if euclidean_norm[i] != 0 and euclidean_norm[j] != 0:
                    # csm_ij = self.vec_product(tf_idf_vectors[i], tf_idf_vectors[j]) / \
                    #          (euclidean_norm[i] * euclidean_norm[j])

                    # csm_ij = cosine_similarity(tf_idf_vectors[i].reshape(1, -1), tf_idf_vectors[j].reshape(1, -1))[0][0]

                    str_i = " ".join(processed_method_list[i])
                    str_j = " ".join(processed_method_list[j])

                    csm_ij = w2v.get_word_sim_score(str_i, str_j)
                    # csm_ij *= self.Wcsm
                    matrix[i][j] += csm_ij
                    matrix[j][i] += csm_ij

        # lsi = LSI()
        # lsi.calc_with_statements(method_list)
        # vm = lsi.calc_lsi()
        # euclidean_norm = lsi.euclidean_norm
        #
        # for i in range(0, num_method):
        #     for j in range(i, num_method):
        #         if i == j:
        #             matrix[i][j] += 1.0
        #             continue
        #         else:
        #             eu = euclidean_norm[i] * euclidean_norm[j]
        #             if eu == 0:
        #                 csm_ij = 0
        #             else:
        #                 csm_ij = self.vec_product(vm[i], vm[j]) / eu
        #             matrix[i][j] = csm_ij
        #             matrix[j][i] = csm_ij



        # print(matrix)
        csm_sum = 0
        for i in range(num_method):
            for j in range(i + 1, num_method):
                csm_sum += matrix[i][j]
        num = (num_method*num_method - num_method) / 2
        c3 = csm_sum / num
        return c3

    def vec_product(self, vector1, vector2):
        prod = 0
        for i in range(0, len(vector1)):
            prod += vector1[i].reshape(-1, 1)*vector2[i]
        return prod

    def get_ATFD(self, pcls):
        atfd_count = 0
        current_field_n_l = []
        all_field_dic = {}
        for cls in self.class_list:
            all_field_n_l = []
            for f in cls.field_list:
                if "public" in f.modifiers and "static" in f. modifiers:
                    all_field_n_l.append(f.field_name)
            all_field_dic[cls.class_name] = all_field_n_l

        print(atfd_count)
        checked_list = []
        for m in pcls.method_list:
            for st in m.statement_list:
                for index, word in enumerate(st.word_list):
                    if word in all_field_dic.keys():
                        if (index + 2) < len(st.word_list):
                            if st.word_list[index+2] in all_field_dic[word]:
                                if word != pcls.class_name:
                                    ck = word + "." + st.word_list[index+2]
                                    if ck in checked_list:
                                        continue
                                    else:
                                        atfd_count += 1
                                        checked_list.append(ck)
        return atfd_count

    # Weighted Method Count (WMC) is the sum of the statical complexity of all methods in a class.
    # We consider McCabe’s cyclo matic complexity metric as a complexity measure.
    def get_WMC(self, cls):
        wmc = 0

        clm = ClassLevelMetrics(sr_class=cls)
        for method in cls.method_list:
            cc = clm.get_method_cc(method)
            wmc += cc
        return wmc

    # The relative number of method pairs of a class that access in common at least one attribute of the measured class
    def get_TCC(self, cls):
        common_method_pair_num = 0
        total_method_pair_num = 0

        num_methods = len(cls.method_list)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                comman_attr = self.get_intersection_il(method1=cls.method_list[i], method2=cls.method_list[j], i_l=cls.field_list)
                if len(comman_attr) > 0:
                    common_method_pair_num += 1
                total_method_pair_num += 1
        if total_method_pair_num == 0:
            return 0
        return round(common_method_pair_num / total_method_pair_num, 2)

    def get_DCC(self, cls):
        dcc_count = 0
        class_name_list = [o.class_name for o in self.class_list]

        for field in cls.field_list:
            if field.field_type in class_name_list:
                dcc_count += 1

        for method in cls.method_list:
            for param in method.param_list:
                if param.type in class_name_list:
                    dcc_count += 1
        return dcc_count

    def get_NOAM(self, cls):
        noam = 0
        f_n_l = [o.field_name.lower() for o in cls.field_list]


        for m in cls.method_list:
            if m.method_name.startswith("get") or m.method_name.startswith("set"):
                m_l = m.method_name.split("get")
                if len(m_l) > 1:
                    if m_l[1].lower() in f_n_l:
                        noam += 1
                else:
                    m_l = m.method_name.split("set")
                    if m_l[1].lower() in f_n_l:
                        noam += 1
        return noam

    def get_CAM(self, cls):
        all_param_l = []
        intersection_param = []

        num_methods = len(cls.method_list)
        for method in cls.method_list:
            for param in method.param_list:
                all_param_l.append(param.type)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                p_l_i = [p.type for p in cls.method_list[i].param_list]
                p_l_j = [p.type for p in cls.method_list[j].param_list]

                i_l = []
                for p in p_l_i:
                    if p in p_l_j:
                        i_l.append(p)
                if len(i_l) > 0:
                    intersection_param.extend(i_l)
        if len(all_param_l) == 0:
            return 0
        return round(len(intersection_param) / len(all_param_l), 2)


    def get_DIT(self, cls):
        count = self.get_parent_count(cls, 0)
        return count

    def get_parent_count(self, sr_class, current_count):
        count = current_count
        if len(sr_class.extends) > 0:
            parent_class_name = sr_class.extends[1]
            parent_cls = None
            count += 1
            for cls in self.class_list:
                if cls.class_name == parent_class_name:
                    parent_cls = cls
            if parent_cls is None:
                return count
            else:
                return self.get_parent_count(parent_cls, count)
        else:
            return count

    def get_intersection_il(self, method1, method2, i_l):
        result = []
        i_l_1 = []
        i_l_2 = []

        i_l_n = [o.field_name for o in i_l]
        for statement in method1.get_all_statement(exclude_special=False):
            for word in statement.word_list:
                if word in i_l_n:
                    i_l_1.append(word)

        for statement in method2.get_all_statement(exclude_special=False):
            for word in statement.word_list:
                if word in i_l_n:
                    i_l_2.append(word)

        if len(i_l_1) > len(i_l_2):
            for w in i_l_1:
                if w in i_l_2:
                    result.append(w)
        else:
            for w in i_l_2:
                if w in i_l_1:
                    result.append(w)

        return result

    def get_merged_LOC(self, cls1, cls2):
        loc1 = self.get_LOC(cls1)
        loc2 = self.get_LOC(cls2)
        return loc1 + loc2

    def get_merged_NOM(self, cls1, cls2):
        nom1 = self.get_NOM(cls1)
        nom2 = self.get_NOM(cls2)
        return nom1 + nom2

    def get_merged_CIS(self, cls1, cls2):
        cis1 = self.get_CIS(cls1)
        cis2 = self.get_CIS(cls2)
        return cis1 + cis2

    def get_merged_NOA(self, cls1, cls2):
        noa1 = self.get_NOA(cls1)
        noa2 = self.get_NOA(cls2)
        return noa1 + noa2

    def get_merged_NOPA(self, cls1, cls2):
        nopa1 = self.get_NOPA(cls1)
        nopa2 = self.get_NOPA(cls2)
        return nopa1 + nopa2

    def get_merged_ATFD(self, cls1, cls2):
        atfd1 = self.get_ATFD(cls1)
        atfd2 = self.get_ATFD(cls2)
        return atfd1 + atfd2

    def get_merged_WMC(self, cls1, cls2):
        wmc1 = self.get_WMC(cls1)
        wmc2 = self.get_WMC(cls2)
        return wmc1 + wmc2

    def get_merged_TCC(self, cls1, cls2):
        common_method_pair_num = 0
        total_method_pair_num = 0
        merged_method_list = []
        merged_field_list = []
        num_methods = 0

        merged_method_list.extend(cls1.method_list)
        merged_field_list.extend(cls1.field_list)
        merged_method_list.extend(cls2.method_list)
        merged_field_list.extend(cls2.field_list)
        num_methods = len(merged_method_list)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                comman_attr = self.get_intersection_il(method1=merged_method_list[i], method2=merged_method_list[j],
                                                       i_l=merged_field_list)
                if len(comman_attr) > 0:
                    common_method_pair_num += 1
                total_method_pair_num += 1
        if total_method_pair_num == 0:
            return 0
        return round(common_method_pair_num / total_method_pair_num, 2)

    def get_merged_LCOM(self, cls1, cls2):
        result = 0

        merged_method_list = []
        merged_field_list = []
        i_l = []
        num_methods = 0

        merged_method_list.extend(cls1.method_list)
        for f in cls1.field_list:
            if "static" not in f.modifiers:
                i_l.append(f)

        merged_method_list.extend(cls2.method_list)
        for f in cls2.field_list:
            if "static" not in f.modifiers:
                i_l.append(f)

        P = 0
        Q = 0
        num_methods = len(merged_method_list)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                intersection_il = self.get_intersection_il(method1=merged_method_list[i],
                                                           method2=merged_method_list[j], i_l=i_l)
                if len(intersection_il) > 0:
                    Q += 1
                else:
                    P += 1
        if P > Q:
            result = P - Q
        else:
            result = 0
        return result

    def get_merged_DCC(self, cls1, cls2):
        dcc1 = self.get_DCC(cls1)
        dcc2 = self.get_DCC(cls2)
        return dcc1 + dcc2

    def get_merged_CAM(self, cls1, cls2):
        all_param_l = []
        intersection_param = []
        merged_method_list = []

        merged_method_list.extend(cls1.method_list)
        for method in cls1.method_list:
            for param in method.param_list:
                all_param_l.append(param.type)

        merged_method_list.extend(cls2.method_list)
        for method in cls2.method_list:
            for param in method.param_list:
                all_param_l.append(param.type)

        num_methods = len(merged_method_list)
        for i in range(0, num_methods):
            for j in range(i, num_methods):
                if i == j:
                    continue
                p_l_i = [p.type for p in merged_method_list[i].param_list]
                p_l_j = [p.type for p in merged_method_list[j].param_list]

                i_l = []
                for p in p_l_i:
                    if p in p_l_j:
                        i_l.append(p)
                if len(i_l) > 0:
                    intersection_param.extend(i_l)
        if len(all_param_l) == 0:
            return 0
        return round(len(intersection_param) / len(all_param_l), 2)

    def get_merged_DIT(self, cls1, cls2):
        dit1 = self.get_DIT(cls1)
        dit2 = self.get_DIT(cls2)

        if dit2 > dit1:
            return dit2
        else:
            return dit1


class MethodLevelMetrics:
    def __init__(self, sr_class):
        self.sr_class = sr_class
        self.field_name_list = []
        self.method_name_list = []

    def get_method_cc(self, sr_method):
        result = 0
        total_statement = sr_method.get_all_statement(exclude_special=False)
        condition_nodes = []
        for st in total_statement:
            if type(st) == SRIFStatement:
                condition_nodes.append(st)
        result = len(condition_nodes) + 1
        return result

    def get_method_loc(self, sr_method):
        result = 0
        result = sr_method.end_line - sr_method.start_line + 1
        return result

    def get_method_pc(self, sr_method):
        result = 0
        result = len(sr_method.param_list)
        return result

    def get_method_LCOM1(self, sr_method):
        fstl = []
        P=0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i+1, len(fstl)):
                if self.check_common_word(
                    l1=list(fstl[i]["var_name_l"]),
                    l2=list(fstl[j]["var_name_l"]),
                ) is False:
                    P += 1


        # print(fstl)
        # print(P)
        return P

    def get_method_LCOM2(self, sr_method):
        fstl = []
        P = 0
        Q = 0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i + 1, len(fstl)):
                if self.check_common_word(
                        l1=list(fstl[i]["var_name_l"]),
                        l2=list(fstl[j]["var_name_l"]),
                ) is False:
                    P += 1
                else:
                    Q += 1

        # print(Q)
        # print(P)
        lcom2 = P - Q

        if lcom2 < 0:
            lcom2 = 0
        return lcom2

    def get_method_LCOM3(self, sr_method):
        fstl = []
        compnent_count = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i + 1, len(fstl)):
                if self.check_common_word(
                        l1=list(fstl[i]["var_name_l"]),
                        l2=list(fstl[j]["var_name_l"]),
                ) is True:
                    if i not in compnent_count:
                        compnent_count.append(i)
                    if j not in compnent_count:
                        compnent_count.append(j)
        return len(compnent_count)

    def get_method_LCOM4(self, sr_method):
        fstl = []
        method_call_count = []
        component_count = 0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            if len(fstl[i]["method_name_l"]) > 0:
                component_count += 1
                for m in fstl[i]["method_name_l"]:
                    if m not in method_call_count:
                        method_call_count.append(m)
                        component_count+=1


        return component_count

    def get_method_LCOM5(self, sr_method):
        fstl = []
        var_used_count = []
        var_total_number = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            if len(fstl[i]["var_name_l"]) > 0:
                for var in fstl[i]["var_name_l"]:
                    var_used_count.append(var)
                    if var not in var_total_number:
                        var_total_number.append(var)
        a = len(var_used_count)
        l = len(var_total_number)
        n = sr_method.get_method_LOC()
        if (l - n * l) == 0:
            return 0
        lcom5 = (a - n * l) / (l - n * l)
        return lcom5

    def get_method_COH(self, sr_method):
        fstl = []
        var_used_count = []
        var_total_number = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            if len(fstl[i]["var_name_l"]) > 0:
                for var in fstl[i]["var_name_l"]:
                    var_used_count.append(var)
                    if var not in var_total_number:
                        var_total_number.append(var)
        a = len(var_used_count)
        l = len(var_total_number)
        n = sr_method.get_method_LOC()
        if (l - n * l) == 0:
            return 0
        lcom5 = (a - n * l) / (l - n * l)
        coh = 1- (1 - 1/n)*lcom5
        return coh

    def get_method_CC(self, sr_method):
        fstl = []
        IVt = 0
        IVc = 0
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for i in range(0, len(fstl)):
            for j in range(i + 1, len(fstl)):
                l1 = list(fstl[i]["var_name_l"])
                l2 = list(fstl[j]["var_name_l"])
                IVt += (len(l1) + len(l2))
                IVc += (len(self.get_common_words(l1, l2)))
        # print("IVt",IVt)
        # print("IVc",IVc)

        n = sr_method.get_method_LOC()
        nt = n-2
        if nt < 0:
            nt = 0
        l = 1*math.factorial(nt)/math.factorial(n)
        r = 0
        rl = math.factorial(n)/(2*math.factorial(nt))

        for i in range(1, int(rl)+1):
            if IVt == 0:
                r += (IVc / 1) * i
                continue
            r += (IVc/IVt)*i
        cc = l * r
        # print("l", l)
        # print("rl", rl)
        # print("r",r)
        return cc

    def get_method_NOAV(self, sr_method):
        fstl = []
        total_var_l = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for obj in fstl:
            for var in obj["var_name_l"]:
                if var not in total_var_l:
                    total_var_l.append(var)
        noav = len(total_var_l) + len(sr_method.param_list)
        return noav

    def get_method_CD(self, sr_method, class_name_list):
        fstl = []
        total_var_l = []
        all_statement = sr_method.get_all_statement(exclude_special=False)
        for st in all_statement:
            method_name_l, var_name_l = self.statement_special_key_filter(st.to_node_word_list())
            object = {
                "method_name_l": method_name_l,
                "var_name_l": var_name_l
            }
            fstl.append(object)

        for obj in fstl:
            for var in obj["var_name_l"]:
                if var in class_name_list:
                    if var not in total_var_l:
                        total_var_l.append(var)
        cd = len(total_var_l)
        return cd

    def get_common_words(self, l1, l2):
        lr = []
        if len(l1) > len(l2):
            for w in l1:
                if w in l2:
                    lr.append(w)
        else:
            for w in l2:
                if w in l1:
                    lr.append(w)

        return lr


    def check_common_word(self, l1, l2):
        if len(l1) > len(l2):
            for w in l1:
                if w in l2:
                    return True
        else:
            for w in l2:
                if w in l1:
                    return True
        return False

    def get_statement_abcl(self, sr_statement):
        result = 0
        a_feat = ["=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "!=", "^=", ">>>=", "++", "--"]

        if type(sr_statement) == SRIFStatement or type(sr_statement) == SRTRYStatement:
            result = 3
        elif type(sr_statement) == SRFORStatement or type(sr_statement) == SRWhileStatement:
            result = 4
        else:
            if "(" in sr_statement.word_list or "new" in sr_statement.word_list:
                result = 2
            else:
                for word in sr_statement.word_list:
                    if word in a_feat:
                        result = 1
                        break
        return result

    def statement_special_key_filter(self, word_list):
        new_st_l = []
        method_name_l = []
        var_name_l = []
        java_keywords = ["boolean", "int", "long", "short", "byte", "float", "double", "char", "class", "interface",
                         "if", "else", "do", "while", "for", "switch", "case", "default", "break", "continue", "return",
                         "try", "catch", "finally", "public", "protected", "private", "final", "void", "static",
                         "strict", "abstract", "transient", "synchronized", "volatile", "native", "package", "import",
                         "throw", "throws", "extends", "implements", "this", "supper", "instanceof", "new", "true",
                         "false", "null", "goto", "const", "=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "!=", "^=", ">>>=", "++", "--", "=="]
        special_key = "[\n`~!@#$%^&*()+=\\-_|{}':;',\\[\\].<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。， 、？]"
        for i, w in enumerate(word_list):
            if w not in java_keywords and w not in special_key and not str(w).isdigit():
                if i < (len(word_list)-1) and word_list[i+1] == "(":
                    method_name_l.append(w)
                else:
                    var_name_l.append(w)
        return method_name_l, var_name_l

    def get_statement_fuc(self, sr_statement):
        result = 0
        if len(self.field_name_list) == 0:
            for field in self.sr_class.field_list:
                self.field_name_list.append(field.field_name)
        for word in sr_statement.to_node_word_list():
            if word in self.field_name_list:
                result += 1
        return result

    def get_method_fuc(self, sr_method):
        result = 0
        for statement in sr_method.statement_list:
            s_re = self.get_statement_fuc(statement)
            result += s_re
        return result

    def get_method_lmuc(self, sr_method):
        result = 0
        for statement in sr_method.statement_list:
            s_re = self.get_statement__lmuc(statement)
            result += s_re
        return result

    def get_statement__lmuc(self, sr_statement):
        result = 0
        if len(self.method_name_list) == 0:
            for m in self.sr_class.method_list:
                self.method_name_list.append(m.method_name)
        for word in sr_statement.to_node_word_list():
            if word in self.method_name_list:
                result += 1
        return result

    def get_statement_puc(self, sr_statement, param_name_list):
        puc = 0
        method_name_l, var_name_l = self.statement_special_key_filter(sr_statement.to_node_word_list())
        for var in var_name_l:
            if var in param_name_list:
                puc+=1
        return puc

    def get_statement_vuc(self, sr_statement):
        method_name_l, var_name_l = self.statement_special_key_filter(sr_statement.to_node_word_list())
        return len(var_name_l)

    def get_statement_wc(self, sr_statement):
        node_word_list = sr_statement.to_node_word_list()
        return len(node_word_list)

    def get_method_block_depth(self, sr_method):
        self.calculate_depth(sr_method=sr_method)
        dp = 0
        for st in sr_method.get_all_statement():
            if st.block_depth > dp:
                dp = st.block_depth
        return dp

    def get_statement_block_depth(self, sr_method, sr_statement):
        if sr_statement.block_depth == -1:
            self.calculate_depth(sr_method=sr_method)
            return sr_statement.block_depth
        else:
            return sr_statement.block_depth

    def calculate_depth(self, sr_method):
        self.__calculate_depth(statement_list=sr_method.statement_list, current_depth=0)

    def __calculate_depth(self, statement_list, current_depth):
        depth = current_depth
        for st in statement_list:
            st.block_depth = depth
            if type(st) == SRIFStatement:
                self.__calculate_depth(statement_list=st.pos_statement_list, current_depth=depth+1)
                self.__calculate_depth(statement_list=st.neg_statement_list, current_depth=depth+1)
            elif type(st) == SRFORStatement:
                self.__calculate_depth(statement_list=st.child_statement_list, current_depth=depth+1)
            elif type(st) == SRWhileStatement:
                self.__calculate_depth(statement_list=st.child_statement_list, current_depth=depth + 1)
            elif type(st) == SRTRYStatement:
                self.__calculate_depth(statement_list=st.try_statement_list, current_depth=depth + 1)
                for cb in st.catch_block_list:
                    self.__calculate_depth(statement_list=cb.child_statement_list, current_depth=depth + 1)
                self.__calculate_depth(statement_list=st.final_block_statement_list, current_depth=depth + 1)
            elif type(st) == SRSwitchStatement:
                for sc in st.switch_case_list:
                    self.__calculate_depth(statement_list=sc.statement_list, current_depth=depth + 1)