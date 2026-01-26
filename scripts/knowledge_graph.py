import os, json
from pprint import pprint
from pyexpat import features
import sys
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
# utils


class KnowledgeGraph:
    def __init__(self, args):
        # graph = nx.MultiGraph() # undirected graph with multi-edges between one node pair
        self.knowledge_graph = nx.Graph()  # undirected graph
        # dictionaries to hold raw data
        self.data_disease = None
        self.data_symptom = None


        # parameter for data loading
        # self.dir_kg = args.knowledge_graph_dir  # use args
        self.dir_kg = './../resources/knowledge_graph/EN'  # hardcoded for now
        self.path_kg_disorder = 'disorder.json'
        self.path_kg_mapping = 'diagnostic_criteria.json'
        self.dir_kg_symptom = os.path.join(self.dir_kg, 'symptom')
        self.path_kg_symptom = os.listdir(self.dir_kg_symptom)
        self.dir_kg_diff_diagnosis = os.path.join(self.dir_kg, 'differential_diagnosis')
        self.path_kg_diff_diagnosis = os.listdir(self.dir_kg_diff_diagnosis)
        self.path_event = 'event.json'

        self.event_mapping = None

        self.load_data()
        self.knowledge_graph = self.build_kg(self.data_disease, self.data_symptom, self.data_mapping, self.data_diff_diagnosis)
        
        # initialize disease features lookup
        self.disease_features_lookup = dict()  # to cache disease features
        for disease_code in self.get_disease_nodes():
            disease_features = self.extract_features_for_disease(disease_code)
            self.disease_features_lookup[disease_code] = disease_features
        print(f"\t\tDisease features lookup initialized for {len(self.disease_features_lookup)} diseases.")

        # parameters for visualization
        self.shape_map = {'disorder': 'o',
                    'symptom_group': 's',
                    'symptom': 'o',
                    'differential_diagnosis': 'D'}
        self.color_map = {'disorder': '#EA526F',
                    'symptom_group': '#227C9D',
                    'symptom': '#B4E1FF',
                    'differential_diagnosis': '#FFCB77'}
        self.node_size_map = {'disorder': 700,
                        'symptom_group': 450,
                        'symptom': 300,
                        'differential_diagnosis': 400}
        self.edge_styles = {
            'must': {'color': '#AA3377', 'style': 'solid', 'width': 2.5, 'rad': 0.12},
            'included_in': {'color': 'black', 'style': 'solid', 'width': 1.2, 'rad': 0.06},
            'differential_diagnosis': {'color': 'black', 'style': 'dotted', 'width': 2.0, 'rad': -0.1},
        }

    def get_kg(self):
        return self.knowledge_graph
    
    def get_disease_nodes(self):
        # dictionary of disease
        # {disease_code: disease_name}
        disease_nodes = {n: d['name'] for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'disorder'}
        return disease_nodes

    def get_symptom_nodes(self):
        # dictionary of symptom
        # {symptom_code: symptom_name}
        symptom_nodes = {n: d['name'] for n, d in self.knowledge_graph.nodes(data=True) if d['type'] == 'symptom'}
        return symptom_nodes

    def get_symptom_info(self, symptom_code):
        if symptom_code not in self.knowledge_graph.nodes:
            raise ValueError(f"Symptom code '{symptom_code}' not found in the knowledge graph.")
        symptom_node = self.knowledge_graph.nodes[symptom_code]
        if symptom_node['type'] != 'symptom':
            raise ValueError(f"Node '{symptom_code}' is not a symptom node.")
        
        symptom_info = self.features_from_symptom_node(symptom_node)
        return symptom_info
    
    def get_differential_diagnosis_nodes(self):
        differential_diagnosis_nodes = {key: val for key, val in self.knowledge_graph.nodes(data=True) if val['type'] == 'differential_diagnosis'}
        return differential_diagnosis_nodes

    def get_differential_diagnosis_info(self, diff_diag_id):
        if diff_diag_id not in self.knowledge_graph.nodes:
            raise ValueError(f"Differential diagnosis ID '{diff_diag_id}' not found in the knowledge graph.")
        diff_diag_node = self.knowledge_graph.nodes[diff_diag_id]
        if diff_diag_node['type'] != 'differential_diagnosis':
            raise ValueError(f"Node '{diff_diag_id}' is not a differential diagnosis node.")
        
        diff_diag_info = {
            'main_disease': self.convert_code_to_name(diff_diag_id.split('-')[0]),
            'comparison_disease': self.convert_code_to_name(diff_diag_id.split('-')[1]),
            'additional_condition': diff_diag_node.get('additional_condition', None),
            'key_difference': diff_diag_node.get('key_difference', None),
            'rules': diff_diag_node.get('rules', None)
        }
        return diff_diag_info

    def get_disease_features_lookup(self):
        return self.disease_features_lookup
    
    def convert_code_to_name(self, code):
        if code not in self.knowledge_graph.nodes:
            raise ValueError(f"Code '{code}' not found in the knowledge graph.")
        node = self.knowledge_graph.nodes[code]
        name = node.get('name', None)
        return name
    
    def convert_name_to_code(self, name):
        for n, d in self.knowledge_graph.nodes(data=True):
            if d["name"].lower() == name.lower():
                return n
        raise ValueError(f"Name '{name}' not found in the knowledge graph.")
    
    def read_json(self, file_path):
        if type(file_path) == str:
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif type(file_path) == list:
            data = {}
            for fp in file_path:
                with open(fp, 'r') as f:
                    obj = json.load(f)
                    data.update(obj)
        return data
    
    def load_data(self):
        ## read json files
        # process to read knowledge graph

        print("\t\t\tReading knowledge graph data files...")
        
        self.data_disease = self.read_json(os.path.join(self.dir_kg, self.path_kg_disorder))
        self.data_symptom = self.read_json([os.path.join(self.dir_kg_symptom, fn) for fn in self.path_kg_symptom])
        self.data_mapping = self.read_json(os.path.join(self.dir_kg, self.path_kg_mapping))
        self.data_diff_diagnosis = self.read_json([os.path.join(self.dir_kg_diff_diagnosis, fn) for fn in self.path_kg_diff_diagnosis])

        self.event_mapping = self.read_json(os.path.join(self.dir_kg, self.path_event))

        print("\t\tData Loaded.")

    def build_kg(self, data_disorder, data_symptom, data_mapping, data_diff_diagnosis):
        print("\t\t\tBuilding knowledge graph...")
        
        graph = self.knowledge_graph
        
        # add disorder nodes
        graph = self.add_disorder_node(graph, data_disorder)

        # add symptom nodes
        graph = self.add_symptom_node(graph, data_symptom)

        # add mapping between disorder and symptom group/symptom
        graph = self.add_mapping_between_disorder_and_symptom(graph, data_mapping)  

        # add differential diagnosis nodes
        graph = self.add_differential_diagnosis_node(graph, data_diff_diagnosis)
        
        
        print(f"\t\tKnowledge graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        # Count nodes and edges by their type
        node_types = nx.get_node_attributes(graph, 'type')
        node_type_counts = {t: list(node_types.values()).count(t) for t in set(node_types.values())}
        print(f"\t\tNode counts by type: {node_type_counts}")

        edge_types = nx.get_edge_attributes(graph, 'relation')
        edge_type_counts = {t: list(edge_types.values()).count(t) for t in set(edge_types.values())}
        print(f"\t\tEdge counts by type: {edge_type_counts}")
        
        return graph
    
    def add_disorder_node(self, graph, data_disorder):
        for disorder_code, disorder_info in data_disorder.items():
            graph.add_node(
                disorder_code, 
                type='disorder',
                label=disorder_code,
                name=disorder_info['name']
            )
        return graph
    
    def add_symptom_node(self, graph, data_symptom):
        for data_symptom_code, symptom_info in data_symptom.items():
            graph.add_node(
                data_symptom_code,
                type='symptom',
                label=data_symptom_code,
                name=symptom_info['name'],
                categoty=symptom_info['category'],
                description=symptom_info['description'],    # detailed description of the symptom
                subtypes=symptom_info.get('subtypes', None),   # dictionary {subsymptom_name: subsymptom_description}
                examples=symptom_info.get('examples', None)   # list of example cases
            )
        return graph

    def add_mapping_between_disorder_and_symptom(self, graph, data_mapping):
        for disorder_code, mapping_info in data_mapping.items():
            # id = disorder_code
            if disorder_code not in graph.nodes:
                print(f"Skipping mapping for unknown disorder code: {disorder_code}")
                continue    # skip if disorder node not in graph
            name = mapping_info['name']
            criteria = mapping_info['required_criteria']

            # update disorder node attributes
            graph.nodes[disorder_code]['additional_requirements'] = None
            graph.nodes[disorder_code]['min_duration'] = None
            graph.nodes[disorder_code]['max_duration'] = None
            graph.nodes[disorder_code]['functional_impairment_required'] = None
            graph.nodes[disorder_code]['psychosocial_stressor_required'] = None
            graph.nodes[disorder_code]['traumatic_stressor_required'] = None

            # add symptom_groups and edges between disorder and symptom nodes
            group_idx = 1
            for _, (criterion, criterion_info) in enumerate(criteria.items()):
                # symptom group node
                if type(criterion_info) is dict:
                    group_name = criterion
                    group_info = criterion_info

                    # add criterion node as symptom group (parent node)
                    group_node_id = group_name.lower()  

                    # symptom group node can be shared across disorders, so check if it already exists
                    if not graph.has_node(group_node_id):
                        graph.add_node(
                            group_node_id,
                            type='symptom_group',
                            label=group_node_id,
                            name=group_name,
                            min_count=group_info['min_count'],  # e.g., minimum number of symptoms to be selected from the symptom pool
                            must_include = "all" if 'must_include_all' in group_info else # type of must_include
                                "one_of" if 'must_include_one_of' in group_info else
                                None,
                            min_duration=group_info.get('min_duration', None),
                            # frequency=group_info.get('frequency', None),  # remove frequency
                            functional_impairment_required = group_info.get('functional_impairment_required', None)
                        )

                    # link disorder to symptom group node
                    relation = 'included_in'    # default relation
                    if 'relation' in group_info:
                        relation_type = group_info['relation']
                        if relation_type == 'must_include':
                            relation = 'must'
                        # else: # relation_type == 'include'
                        
                    graph.add_edge(
                        disorder_code, group_node_id,
                        relation=relation
                    )


                    # link symptoms to the criterion node
                    symptom_pool = group_info.get('symptom_pool', [])
                    
                    must_include_symptoms = []
                    if 'must_include_one_of' in group_info:
                        must_include_symptoms = group_info['must_include_one_of']
                    elif 'must_include_all' in group_info:
                        must_include_symptoms = group_info['must_include_all']
                    else:
                        pass    # must_include_symptoms remains empty

                    symptom_pool = list(set(symptom_pool + must_include_symptoms))  # ensure must_include symptoms are also in the pool
                    for symptom_code in symptom_pool:
                        graph.add_edge(
                            group_node_id, symptom_code,
                            relation='included_in',   # e.g., 'included_in'
                        )

                    # link must edge for symptoms that must be included
                    for must_symptom_code in must_include_symptoms:
                        if graph.has_edge(group_node_id, must_symptom_code):
                            # update existing edge relation to 'must' instead of adding a new edge
                            graph[group_node_id][must_symptom_code]['relation'] = 'must'
                        else:
                            # if edge doesn't exist for some reason, add it
                            graph.add_edge(
                                group_node_id, must_symptom_code,
                                relation='must'
                            )
                    group_idx += 1
                # other attributes
                else:
                    # update disorder node attributes
                    graph.nodes[disorder_code][criterion] = criterion_info 
        return graph

    def add_differential_diagnosis_node(self, graph, data_diff_diagnosis):
        for disease_code, diff_disease_list in data_diff_diagnosis.items():
            for diff_diag_info in diff_disease_list:
                diff_disease_code = diff_diag_info['differential_diagnosis']
                additional_condition = diff_diag_info.get('additional_condition', None)
                key_difference = diff_diag_info['key_difference']
                rules = diff_diag_info['rules']

                if disease_code not in graph.nodes or diff_disease_code not in graph.nodes:
                    print(f"Skipping differential diagnosis between {disease_code} and {diff_disease_code} as one of the diseases is not in the graph.")
                    continue    # skip if either disease node is not in the graph
                
                diff_diag_id = "-".join([disease_code, diff_disease_code])  # directed id
                
                if graph.has_node(diff_diag_id):
                    print(f"Skipping adding duplicate differential diagnosis node: {diff_diag_id}")

                try:
                    graph.add_node(
                        diff_diag_id,
                        type='differential_diagnosis',
                        label=diff_diag_id,
                        name=f"DiffDiag:{self.convert_code_to_name(disease_code)}-{self.convert_code_to_name(diff_disease_code)}",
                        additional_condition=additional_condition,
                        key_difference=key_difference,
                        rules={
                            disease_code: rules[disease_code],
                            diff_disease_code: rules[diff_disease_code]
                        }
                    )
                except Exception as e:
                    print(f"Error adding differential diagnosis node with Error {e}")
                    print(f"Diff Diagnosis ID: {disease_code} / Comparison between {diff_disease_code}")
                    continue

                # connect differential diagnosis node to both disorder nodes            
                graph.add_edge(
                    disease_code, diff_diag_id,
                    relation='differential_diagnosis'
                )
                graph.add_edge(
                    diff_disease_code, diff_diag_id,
                    relation='differential_diagnosis'
                )
        return graph


    def retrieve_disease_info(self, disease):
        if self.data_disease is None:
            raise ValueError("Disease data is not loaded.")
        
        # find disease code and name
        if disease in self.data_disease:
            # direct lookup by code
            disease_code = disease
            disease_name = self.data_disease[disease_code]['name']
        elif disease.lower() in [info['name'].lower() for info in self.data_disease.values()]:
            # lookup by disease name
            for code, info in self.data_disease.items():
                if info['name'].lower() == disease.lower():
                    disease_code = code
                    disease_name = info['name']
                    break
        else:
            raise ValueError(f"Disease '{disease}' not found in loaded data.")
        
        # retrieve disease info from the knowledge graph
        if disease_code not in self.knowledge_graph.nodes:
            raise ValueError(f"Disease code '{disease_code}' not found in the knowledge graph.")
        
        # reconstruct disease info
        disease_info = {}   
        
        disease_node = self.knowledge_graph.nodes[disease_code]
        keys = ['name', 'label', 
                'additional_requirements',
                'min_duration', 'max_duration',
                'functional_impairment_required', 'psychosocial_stressor_required', 'traumatic_stressor_required']
        disease_info['code'] = disease_code
        disease_info['name'] = disease_name
        for key in keys:
            value = disease_node.get(key, None)
            if value is not None:
                disease_info[key] = value

        symptom_groups = []
        for neighbor in self.knowledge_graph.neighbors(disease_code):
            if self.knowledge_graph.nodes[neighbor]['type'] == 'symptom_group':
                symptom_groups.append((neighbor, self.knowledge_graph.nodes[neighbor]))

        disease_info['symptom_groups'] = {}
        symptom_groups = sorted(symptom_groups)
        for sg, sg_info in symptom_groups:
            disease_info['symptom_groups'][sg] = {
                k:v for k, v in sg_info.items()
            }
            
            # collect symptoms linked to this symptom group
            symptoms = []
            for neighbor in self.knowledge_graph.neighbors(sg):
                try:
                    if self.knowledge_graph.nodes[neighbor]['type'] == 'symptom':
                        symptoms.append((neighbor, self.knowledge_graph.nodes[neighbor]))
                except Exception as e:
                    print(f"Error processing symptom node {neighbor}: {e}")
                    exit(1)
            disease_info['symptom_groups'][sg]['symptoms'] = {sym_code: sym_info for sym_code, sym_info in sorted(symptoms)}
        
        return disease_info

    def show_disease_info_pretty(self, disease_info):
        # import pprint
        #pp = pprint.PrettyPrinter(indent=2, )
        #pp.pprint(disease_info)

        data_json = json.dumps(disease_info, indent=4, ensure_ascii=False)
        print(data_json)
        
        return data_json
    
    
    def extract_disease_from_differential_diagnosis(self, diff_diag_id):
        # 
        # input: differential diagnosis node id (e.g., "D001-D002")
        # output: dictionary of two disease features, diagnostic feature, and common symptom pool
        #
    
        if self.knowledge_graph is None or diff_diag_id not in self.knowledge_graph.nodes:
            raise ValueError(f"Differential diagnosis ID '{diff_diag_id}' not found in the knowledge graph.")
        
        disease_codes = diff_diag_id.split('-')
        if len(disease_codes) != 2:
            raise ValueError(f"Differential diagnosis ID is not in the expected format: {diff_diag_id}")
        
        return disease_codes
    
    
    def extract_features_for_disease(self, disease_code):
        if disease_code not in self.knowledge_graph.nodes:
            raise ValueError(f"Disease code '{disease_code}' not found in the knowledge graph.")
        elif self.knowledge_graph.nodes[disease_code]['type'] != 'disorder':
            raise ValueError(f"Node '{disease_code}' is not a disorder node.")
    
        disease_node = self.knowledge_graph.nodes[disease_code]
        
        disease_name = disease_node.get('name', None)

        # diagnostic criteria attributes
        diagnostic_criteria = self.features_from_disease_node(disease_node)
        
        # symptom groups and symptoms
        symptom_groups_info = {'must_include_groups': {}, 'optional_groups': {}}
        # iterate over symptom groups linked to the disease
        for neighbor in self.knowledge_graph.neighbors(disease_code):
            if self.knowledge_graph.nodes[neighbor]['type'] == 'symptom_group':
                symptom_group_node = self.knowledge_graph.nodes[neighbor] # id: neighbor
                relation_type = self.knowledge_graph[disease_code][neighbor]['relation']
                if relation_type == 'must':
                    symptom_groups_info['must_include_groups'][neighbor] = self.features_from_symptom_group_node(symptom_group_node)
                else:
                    symptom_groups_info['optional_groups'][neighbor] = self.features_from_symptom_group_node(symptom_group_node)
                
                must_include_symptoms = []
                include_symptoms = []
                for sg_neighbor in self.knowledge_graph.neighbors(neighbor):
                    if self.knowledge_graph.nodes[sg_neighbor]['type'] == 'symptom':
                        if self.knowledge_graph[neighbor][sg_neighbor]['relation'] == 'must':
                            must_include_symptoms.append(sg_neighbor)
                        else:
                            include_symptoms.append(sg_neighbor)
                
                if relation_type == 'must':
                    symptom_groups_info['must_include_groups'][neighbor]['must_include_symptoms'] = must_include_symptoms
                    symptom_groups_info['must_include_groups'][neighbor]['include_symptoms'] = include_symptoms
                else:
                    symptom_groups_info['optional_groups'][neighbor]['must_include_symptoms'] = must_include_symptoms
                    symptom_groups_info['optional_groups'][neighbor]['include_symptoms'] = include_symptoms
        
        full_symptom_pool = set()
        for group_type in symptom_groups_info:
            for sg_info in symptom_groups_info[group_type].values():
                full_symptom_pool.update(sg_info.get('must_include_symptoms', []))
                full_symptom_pool.update(sg_info.get('include_symptoms', []))
        full_symptom_pool = sorted(list(full_symptom_pool))

        return {
            "code": disease_code,
            "name": disease_name,
            "diagnostic_criteria": diagnostic_criteria,
            "symptom_groups": symptom_groups_info,
            "full_symptom_pool": full_symptom_pool
        }

    def features_from_disease_node(self, disease_node):
        # min_duration, max_duration, additional_requirements, functional_impairment_required, psychosocial_stressor_required, traumatic_stressor_required
        features = {}
        keys = disease_node.keys()
        exclude = ['name', 'type', 'label']
        keys = [k for k in disease_node.keys() if k not in exclude]
        for diag_key in keys:
            features[diag_key] = disease_node.get(diag_key, None)
        return features
    
    def features_from_symptom_group_node(self, symptom_group_node):
        # symptom group attributes: min_count, must_include, min_duration, functional_impairment_required
        features = {}
        keys = symptom_group_node.keys()
        exclude = ['type', 'label']
        keys = [k for k in symptom_group_node.keys() if k not in exclude]
        for key in keys:
            features[key] = symptom_group_node.get(key, None)
        return features
    
    def features_from_symptom_node(self, symptom_node):
        # symptom attributes: category, description, subtypes, examples
        features = {}
        keys = symptom_node.keys()
        exclude = ['type', 'label']
        keys = [k for k in symptom_node.keys() if k not in exclude]
        for key in keys:
            features[key] = symptom_node.get(key, None)
        return features
    
    def get_event_mapping_dictionary(self, event_key):
        if self.event_mapping is None:
            raise ValueError("Event mapping data is not loaded.")
        if event_key not in self.event_mapping:
            raise ValueError(f"Event key '{event_key}' not found in event mapping data.")

        return self.event_mapping[event_key]


    def get_all_keys_in_kg(self):
        if self.knowledge_graph is None:
            raise ValueError("Knowledge graph is not loaded.")
        
        all_keys = dict()

        for _, node_data in self.knowledge_graph.nodes(data=True):
            node_type = node_data.get('type', None)
            if node_type not in all_keys:
                all_keys[node_type] = dict()
            for key, val in node_data.items():
                if key not in all_keys[node_type]:
                    all_keys[node_type][key] = set()
                all_keys[node_type][key].add(type(val).__name__)
        
        return {k: {key: sorted(list(types)) for key, types in v.items()} for k, v in all_keys.items()}

    def get_all_keys_in_disease_features(self):
        all_keys = dict()

        all_keys = defaultdict(set)

        
        for features in self.disease_features_lookup.values():
            _traverse(features)

        all_keys = dict(all_keys)
        
        return {key: sorted(list(types)) for key, types in all_keys.items()}
    
    def visualize_graph(self):
        print("\t\t\tVisualizing knowledge graph...")

        if self.knowledge_graph is None or self.knowledge_graph.number_of_nodes() == 0:
            raise ValueError("Knowledge graph is not loaded or is empty.")
        
        graph = self.knowledge_graph
        
        shape_map = self.shape_map
        color_map = self.color_map
        node_size_map = self.node_size_map
        edge_styles = self.edge_styles
    
        # larger figure + spring layout with bigger spacing (k) and more iterations for stable layout
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(graph, k=0.8, iterations=200, seed=42)

        labels = nx.get_node_attributes(graph, 'label')
        
        # draw edges first (under nodes) so nodes/labels remain visible
        ## draw relation-specific edges
        ## collect edges with data (normalize to (u, v, data) tuples)
        # raw_edges = [(u, v, d) for u, v, k, d in graph.edges(keys=True, data=True)]   # support nx.MultiGraph
        raw_edges = list(graph.edges(data=True))    # support nx.Graph

        for rel, cfg in edge_styles.items():
            edgelist = [(u, v) for u, v, d in raw_edges if d.get('relation') == rel]
            if edgelist:
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=edgelist,
                    edge_color=cfg['color'],
                    style=cfg['style'],
                    width=cfg['width'],
                    alpha=0.9,
                    connectionstyle=f"arc3,rad={cfg['rad']}",
                    arrows=True
                )

        ## any other edges in light gray
        other = [(u, v) for u, v, d in raw_edges if d.get('relation') not in edge_styles]
        if other:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=other,
                edge_color='#999999',
                style='dashed',
                width=0.9,
                alpha=0.6,
                connectionstyle="arc3,rad=0.03",
            )

        # draw nodes by node type
        ## group nodes by their 'type' attribute and draw each group with a different shape/color
        types = nx.get_node_attributes(graph, 'type')
        nodes_by_type = {}
        for n, t in types.items():
            nodes_by_type.setdefault(t, []).append(n)

        for t, nodes in nodes_by_type.items():
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=nodes,
                node_shape=shape_map.get(t, 'o'),
                node_color=color_map.get(t, 'green'),
                node_size=node_size_map.get(t, 400),
                linewidths=1,
                edgecolors='k',
                alpha=0.95,
                # zorder=2
            )

        # labels with a light bbox to improve readability over edges
        nx.draw_networkx_labels(
            graph, pos, labels=labels,
            font_size=7,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 0.1},
            # zorder=3
        )

        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./knowledge_graph_visualization.png', dpi=300)
        plt.show()



if __name__ == "__main__":
    args = None
    kg_obj = KnowledgeGraph(args)
    kg = kg_obj.get_kg()
    kg_obj.visualize_graph()
    
    # save_dir = './../resources/knowledge_graph/KR/readable_kg_for_validation'
    # os.makedirs(save_dir, exist_ok=True)

    # export disease node in graph to readable json files
    # disease_nodes = kg_obj.get_disease_nodes()
    # symptom_nodes = kg_obj.get_symptom_nodes()
    # with open(os.path.join(save_dir, 'disease_nodes.json'), 'w', encoding='utf-8') as f:
    #     json.dump(disease_nodes, f, indent=4, ensure_ascii=False)
    # with open(os.path.join(save_dir, 'symptom_nodes.json'), 'w', encoding='utf-8') as f:
    #     json.dump(symptom_nodes, f, indent=4, ensure_ascii=False)

    # export each disease info to a readable json file
    ## validation purpose
    # for code, name in disease_nodes.items():
    #     info = kg_obj.retrieve_disease_info(code)
    #     dt = kg_obj.show_disease_info_pretty(info)
    #     save_path = f'./../resources/knowledge_graph/KR/readable_kg_for_validation/{code}_{name.replace("/", "-")}.json'
    #     with open(save_path, 'w', encoding='utf-8') as f:
    #         f.write(dt)

    # test extraction of features from disease node
    # test_disease_code = 'D001'
    d1_info = kg_obj.extract_features_for_disease('D002')
    pprint(d1_info)

    # test differential diagnosis extraction
    diff_diag_id = 'D001-D008'
    disease_codes = kg_obj.extract_disease_from_differential_diagnosis(diff_diag_id)
    for dc in disease_codes:
        dc_info = kg_obj.extract_features_for_disease(dc)
        pprint(dc_info)

    # diff_diag_id = 'D001-D023'  # doesnt exist
    # disease_codes = kg_obj.extract_disease_from_differential_diagnosis(diff_diag_id)
    # for dc in disease_codes:
    #     dc_info = kg_obj.extract_features_for_disease(dc)
    #     pprint(dc_info)