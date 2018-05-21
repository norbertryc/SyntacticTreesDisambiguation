import xml.etree.ElementTree as ET
from copy import deepcopy
import numpy as np


def is_forest_correct(xml_tree):

    
    if xml_tree.getroot().tag != "forest":
        return False

    base_answer_type = xml_tree.getroot().find('.//answer-data//base-answer').attrib["type"]
    
    if base_answer_type != "FULL":
        return False

    return True
    


def _check_sentence(xml_tree, accept_tags=["forest","tree"]):
    
    """
    Funkcja sprawdza poprawnosc wypowiedzenia i arumentu: 
    - czy istnieje dla niego poprawne drzewo - wypowiedzenie jest poprawne jesli base_answer na polu "type" ma wartosc "FULL".
    - arumentem powinno byc drzewo o tagu korzenia rownym "forest" lub "tree".
    [W oryginalnych plikach z lasami jest to "forest", natomiast gdy z lasu tworzone sa pojedyncze drzewa,
    to maja one tag "tree"]
    
    xml_tree - las drzew lub drzewo [xml.etree.ElementTree.ElementTree]
    """
    
    if type(xml_tree) != ET.ElementTree:
        raise AssertionError("Argument xml_tree is not not ElementTree")
    
    
    if type(accept_tags) == str:
        accept_tags = [accept_tags]
    
    
    if not xml_tree.getroot().tag in accept_tags:
        raise AssertionError('Argument in not in [' + ",".join(accept_tags) + '] - it has tag "' + xml_tree.getroot().tag + '"' )
    
    
    base_answer_type = xml_tree.getroot().find('.//answer-data//base-answer').attrib["type"]
    correct = base_answer_type == "FULL"

    if not correct:
        raise AssertionError("Sentence is not correct: Node <base-answer> has type value " + base_answer_type  + " instead of 'FULL'")
        
    pass


def get_random_tree(forest, random_state=None):
    
    """
    Funkcja zwraca losowe drzewo z upakowanego lasu (forest).
    Dla lasu, w ktorym nie ma poprawnego drzewa funkcja wyrzuca blad.
    
    forest - las drzew [xml.etree.ElementTree.ElementTree]
    """

    # sprawdzenie poprawnosci lasu i ewentualne wypisanie komunikatu
    _check_sentence(forest,"forest")
    
    # ustawiamy ziarno
    if random_state is not None:
        np.random.seed(random_state)
            
            
    root_old = forest.getroot()
    root_new = ET.Element("tree",root_old.attrib)
    
    
    # las sklada sie z drzew (wezly "node") oraz dodatkowych danych (inne wezly) -
    # tresc wypowiedzenia, statystyki lasu, itd. - i tutaj przepisujemy te wezly
    features = root_old.getchildren()
    for feature in features:
        if feature.tag != "node": 
            feature_copy = deepcopy(feature)
            if feature_copy.tag == "stats":
                feature_copy.tag = "forest-stats"
                
            root_new.append(feature_copy) # modyfikujemy tag wezla wiec potrzebna kopia, zeby nie zmodyfikowac oryginalnego drzewa
    
    # definiujemy wezel ze statystykami drzewa
    # robimy to w tym iejscu zeby zachowac logiczna kolejnosc wezlow - zeby wypisywalo sie to na poczatku
    # wartosci nadamy nizej
    ET.SubElement(root_new, "tree-stats", {"height":"0","nodes":"0"})
            
            
    # definiujemy rekurencyjna funkcje, ktora bedzie przechodzic po lesie i
    # kolekcjonowac wezly, tworzac losowe drzewo.
    # drzewo jest tworzone na korzeniu root_new.
    def add_random_children(current_node_old):
        
        current_node_new = ET.SubElement(root_new, current_node_old.tag, current_node_old.attrib)
        
        features = current_node_old.getchildren()
        # kazdy "node" jest terminalem albo nieterminalem i ma opis wlasnosci
        # i tutaj wyciagamy te wlasnosci z wezla innego niz "children"
        for feature in features:
            if feature.tag != "children": 
                current_node_new.append(feature)
        
        children_old = current_node_old.findall("children")
        if len(children_old) == 0: #jestesmy w lisciu wiec konczymy dzialanie funkcji
            return None
        random_children_old = children_old[np.random.choice(len(children_old),1)[0]]
        random_children_new = ET.SubElement(current_node_new, random_children_old.tag, random_children_old.attrib)
        for child_old in random_children_old.getchildren():
            x = ET.SubElement(random_children_new, child_old.tag, child_old.attrib)
            next_node = root_old.find('.//node[@nid="' + x.attrib["nid"] + '"]')
            add_random_children(next_node)
        
    
        # wezel startowy (przyjmujemy, ze node z id=0 jest zawsze pierwszy):
    # TODO: upewnic sie czy to jest poprawne podejscie - czy moze byc inny wezel poczatkowym
    node_0 = root_old.find('.//node[@nid="0"]') 
    
    # konstruujemy drzewo:
    add_random_children(node_0)
    
    new_tree = ET.ElementTree(root_new)
    
    th = _tree_height(new_tree, node_id=0)
    
    root_new.find("tree-stats").attrib["height"] = str(th)
    root_new.find("tree-stats").attrib["nodes"] = str(len(root_new.findall("node")))
    
    return new_tree
       
 

def number_of_trees_in_forest(forest):

    """
    Funkcja zwraca liczbe drzew w lesie forest.
    
    forest - las drzew [xml.etree.ElementTree.ElementTree]
    """
    
    _check_sentence(forest,"forest")
    
    return int(forest.find("stats").attrib["trees"])
    
    
def get_random_negative_tree(forest, random_state=None):
    
    """
    Funkcja zwraca losowe negatywne (niepoprawne) drzewo z lasu forest.
    
    Gdy las sklada sie tylko z jednego drzewa (poprawnego) to zwracana jest wartosc None.
    
    forest - las drzew [xml.etree.ElementTree.ElementTree]
    """
    
    _check_sentence(forest,"forest")
    
    
    number_of_trees = number_of_trees_in_forest(forest)
    
    if number_of_trees == 1:
        Warning("There is only one tree in the forest")
        return None
    
    else:
        while True:
            tree = get_random_tree(forest,random_state)
            if not is_positive(tree):
                return tree
    
    


def terminals(tree):

    terminal_nodes = [x for x in tree.findall("node[terminal]")]

    terminals = [[(x.attrib["nid"],
                   x.find("terminal//orth").text.replace(" ", ""),  # zdarzaja sie przypadki ze token zawiera w sobie spacje i potem wyglada to jakby bylo wiecej tokenow i sie dlugosc nie zgadza
                   x.find("terminal//base").text, 
                   x.find("terminal//f").text)]  for x in terminal_nodes]

    ids = [x[0][0] for x in terminals]

    return terminals, ids 



def dependency_tree(tree):

    dep_tree, ids = terminals(tree)
    n_terminals = len(dep_tree)
    
    for nid in ids:

        parent = tree.find(".//children/child[@nid='"+str(nid)+"']....")

        if parent is not None:
            loc =  np.where([str(nid) in [x[0] for x in branch] and len(branch[-1])>=2 for branch in dep_tree])[0]

            if parent.attrib["nid"] not in ids:
                ids.append(parent.attrib["nid"])


            if len(parent.findall("children/child"))==1:
                
                dep_tree[loc[0]].append(tuple([parent.attrib["nid"]] +[x.text for x in parent.find("nonterminal").getchildren()]))

                if parent.attrib["nid"] == "0":
                    
                    if parent.attrib["nid"] not in [branch[0][0] for branch in dep_tree]:
                        dep_tree.append([tuple([parent.attrib["nid"]] +[x.text for x in parent.find("nonterminal").getchildren()])])
   
            
            else:
                
                dep_tree[loc[0]].append((parent.attrib["nid"],))
                
                if parent.attrib["nid"] not in [branch[0][0] for branch in dep_tree]:
                    dep_tree.append([tuple([parent.attrib["nid"]] +[x.text for x in parent.find("nonterminal").getchildren()])])
        else:
            pass
            #labels.append(get_subtree_label(tree, tree.find(".//node[@nid='" + str(nid) + "']")))
            
    heads = [x[-2][0] if len(x)>1 else x[0][0]  for x in dep_tree]       
    labels = [get_subtree_label(tree, tree.find(".//node[@nid='" + str(nid) + "']")) for nid in heads]

    return(dep_tree, labels, n_terminals)                                           



def get_head(tree, node_id):
    """
    Funkcja zwraca słowo, które byłoby korzeniem poddrzewa drzewa 'tree' wychodzącego z wierzchołka 'node_id' w rozbiorze zależnościowym.
    """
    
    node = tree.find(".//node[@nid='" + str(node_id) + "']")

    while not ( len(node.getchildren())==1 and node.getchildren()[0].tag=="terminal"):

        try:
            head_child_id = node.find("children/child[@head='true']").attrib["nid"]
            node = tree.find(".//node[@nid='" + str(head_child_id) + "']")
        except:
            return("__head_unknown__")
        
    head = node.find("terminal//orth").text.replace(" ", "")
    
    return(head)


def transform_to_dependency_format(tree):
    
    dep_tree, labels, n_terminals = dependency_tree(tree)
    
    values = [[x[1] for x in branch[:-1]] for branch in dep_tree]
    
    top_node_ids = [branch[-1][0] for branch in dep_tree]
    
    for i in range(n_terminals,len(values)):
        if len(values[i])>0:
            values[i] = [get_head(tree, top_node_ids[i])]+values[i]
        else:
            values[i] = [get_head(tree, top_node_ids[i]),"__wypowiedzenie__"]


    tokens_and_rules = [(y[0],"-".join(y[1:])) if len(y)>1 else (y[0],"__brak__") for y in values]

    nodes_ids = [[x[0] for x in branch] for branch in dep_tree]
    
    parent_ids = [0]*len(nodes_ids)
    firsts = [x[0] for x in nodes_ids]


    for i in range(len(nodes_ids)):
        last = nodes_ids[i][-1]

        if len(nodes_ids[i])==1 and last == "0":
            parent_ids[i] = 0
        else:
            parent_ids[i] = np.where([last == x for x in firsts])[0][0] + 1 # "+1" po to zeby format danych zgadzal sie z tymi ze stanfordu 
                                                                            # - numerujemy tokeny od 1, a nie od 0

    nodes_used_in_tree = [x[0] for branch in dep_tree for x in branch]    
    
    dependency_data = list(zip([x[0] for x in tokens_and_rules],[x[1] for x in tokens_and_rules], parent_ids, labels))
    
    return(dependency_data, nodes_used_in_tree)






def write_dependency_format(dep_tree, folder, overwrite=False):
    
    if not overwrite:
        mode = "a+"
    else:
        mode = "w"
    
    tokens = [x[0] for x in dep_tree[0]]
    with open(folder+"/tokens.txt", mode) as f:
        f.write(" ".join(tokens) + "\n")
        
    rules = [x[1] for x in dep_tree[0]]
    with open(folder+"/rules.txt", mode) as f:
        f.write(" ".join(rules) + "\n")
        
    parents = [str(x[2]) for x in dep_tree[0]]
    with open(folder+"/parents.txt", mode) as f:
        f.write(" ".join(parents) + "\n")
        
    labels = [str(x[3]) for x in dep_tree[0]]
    with open(folder+"/labels.txt", mode) as f:
        f.write(" ".join(labels) + "\n")

    nodes_used_in_tree = dep_tree[1]
    with open(folder+"/nodes_used_in_tree.txt", mode) as f:
        f.write(" ".join(nodes_used_in_tree) + "\n")



def _tree_height(xml_tree, node_id=0):
    
    """
    Funkcja oblicza wysokosc drzewa (dlugosc najdluzszej sciezki od korzenia do liscia)
    lub lasu (maximum z wszystkich mozliwych drzew)
    
    xml_tree - drzewo luba las drzew lub korzen drzewa jednego lub drugiego
    """
    
    
    if type(xml_tree)==ET.Element:
        node = xml_tree
    else:
        node = xml_tree.getroot()
        
    node = node.find('.//node[@nid="' + str(node_id) + '"]')
    children = node.findall(".//children//child")
    
    if len(children)==0:
        return 1
    else:
        children_nodes = [child.attrib["nid"] for child in children]
        return 1+max([_tree_height(xml_tree,x) for x in children_nodes])
        



def number_of_nodes(tree):
    """
    Zwraca liczbe wezlow w drzewie.
    
    tree - drzewo lub korzen drzewa
    """
    if type(tree)==ET.Element:
        return len(tree.findall("node"))
    else:
        return len(tree.getroot().findall("node")) 




def get_subtree_label(tree, node):
    
    if node.find("children") is None:
        return 1
    
    if node.find("children").attrib.get("chosen","false") == "false":
        return 0
    else:
        return int(np.all([get_subtree_label(tree, tree.find(".//node[@nid='"+ x.attrib["nid"] + "']")) for x in node.find("children").findall("child")]))
        


def is_positive(tree): 
    
    """
    Funkcja sprawdza czy drzewo jest pozytywne - czy jest poprawnym drzewem rozbioru
    Zwraca wartosc logiczna.
    
    tree - drzewo [xml.etree.ElementTree.ElementTree]
    """
    
    _check_sentence(tree,"tree")
    
    assert len(tree.find("node"))>0, 'There is not "node" element in the tree'
    
    #Sprawdzamy czy wszystkie wezly "node" maja wartosc chosen="true":
    for x in tree.iter("node"):
        if not x.attrib["chosen"]=="true":
            return False

    #Sprawdzamy czy wszystkie wezly "children" maja wartosc chosen="true":
    for x in tree.iter(".//children"):
        if not x.attrib["chosen"]=="true":
            return False
        
    return True
        
        
    


def get_positive_tree(forest):
    
    """
    Funkcja zwraca poprawne (pozytywne) drzewo z upakowanego lasu (forest).
    Dla lasu, w ktorym nie ma poprawnego drzewa funkcja wyrzuca blad.
    
    forest - las drzew [xml.etree.ElementTree.ElementTree]
    """

    # sprawdzenie poprawnosci lasu i ewentualne wypisanie komunikatu
    _check_sentence(forest,"forest")
            
    root_old = forest.getroot()
    root_new = ET.Element("tree",root_old.attrib)
    
    
    # las sklada sie z drzew (wezly "node") oraz dodatkowych danych (inne wezly) -
    # tresc wypowiedzenia, statystyki lasu, itd. - i tutaj przepisujemy te wezly
    features = root_old.getchildren()
    for feature in features:
        if feature.tag != "node": 
            feature_copy = deepcopy(feature)
            if feature_copy.tag == "stats":
                feature_copy.tag = "forest-stats"
                
            root_new.append(feature_copy) # modyfikujemy tag wezla wiec potrzebna kopia, zeby nie zmodyfikowac oryginalnego drzewa
            
    # definiujemy rekurencyjna funkcje, ktora bedzie przechodzic po lesie i
    # kolekcjonowac wezly, tworzac losowe drzewo.
    # drzewo jest tworzone na korzeniu root_new.
    def add_positive_children(current_node_old):
        
        current_node_new = ET.SubElement(root_new, current_node_old.tag, current_node_old.attrib)
        
        features = current_node_old.getchildren()
        # kazdy "node" jest terminalem albo nieterminalem i ma opis wlasnosci
        # i tutaj wyciagamy te wlasnosci z wezla innego niz "children"
        for feature in features:
            if feature.tag != "children": 
                current_node_new.append(feature)
        
        
        children_old = current_node_old.findall('children[@chosen="true"]')
        # powinno byc tylko jedno takie dziecko
        
        assert len(children_old) <= 1, 'More than one children has chosen="true"'
        
        if len(children_old) == 0: #jestesmy w lisciu wiec konczymy dzialanie funkcji
            return None
        
        #random_children_old = children_old[np.random.choice(len(children_old),1)[0]]
        children_new = ET.SubElement(current_node_new, children_old[0].tag, children_old[0].attrib)
        for child_old in children_old[0].getchildren():
            x = ET.SubElement(children_new, child_old.tag, child_old.attrib)
            next_node = root_old.find('.//node[@nid="' + x.attrib["nid"] + '"]')
            assert next_node.attrib["chosen"] == "true"
            add_positive_children(next_node)
        
    
    # wezel startowy (przyjmujemy, ze node z id=0 jest zawsze pierwszy):
    # TODO: upewnic sie czy to jest poprawne podejscie - czy moze byc inny wezel poczatkowym
    node_0 = root_old.find('.//node[@nid="0"][@chosen="true"]') 
    
    # konstruujemy drzewo:
    add_positive_children(node_0)
    
    positive_tree = ET.ElementTree(root_new)

    # Sprawdzenie poprawnosci drzewa
    assert is_positive(positive_tree), """Something gone wrong - tree is not positive"""
        
        
    return positive_tree
    


def get_n_negative_trees_randomly(forest, n = "all", random_state=None):

    """
    Funkcja zwraca losowe negatywne (niepoprawne) drzewo z lasu forest.

    Gdy las sklada sie tylko z jednego drzewa (poprawnego) to zwracana jest wartosc None.

    forest - las drzew [xml.etree.ElementTree.ElementTree]
    """

    _check_sentence(forest,"forest")

    if n == "all":
        n = number_of_trees_in_forest(forest)-1
    
    assert n <= number_of_trees_in_forest(forest)-1, "You want to get more trees than there is in the forest."

    

    if n == 1:
        Warning("There is only one tree in the forest and it i spositive")
        return None

    else:
        ids = set()#"-".join(sorted([node.attrib["nid"] for node in trees[0].findall('node')])))
        trees = []
        while n:
            tree = get_random_tree(forest,random_state)
            key = "-".join(sorted([node.attrib["nid"] for node in tree.findall('node')]))
            if (not is_positive(tree)) and (id not in ids):
                trees.append(tree)
                ids.add(key)
                n -= 1
    
    return trees



def get_all_trees_randomly(forest, random_state=None):

    """
    Funkcja zwraca losowe negatywne (niepoprawne) drzewo z lasu forest.

    Gdy las sklada sie tylko z jednego drzewa (poprawnego) to zwracana jest wartosc None.

    forest - las drzew [xml.etree.ElementTree.ElementTree]
    """

    _check_sentence(forest,"forest")

    n = number_of_trees_in_forest(forest)

    if n == 1:
        Warning("There is only one tree in the forest")
        return [get_positive_tree(forest)]

    else:
        trees = [get_positive_tree(forest)]
        ids = set("-".join(sorted([node.attrib["nid"] for node in trees[0].findall('node')])))
        while n:
            tree = get_random_tree(forest,random_state)
            key = "-".join(sorted([node.attrib["nid"] for node in tree.findall('node')]))
            if id not in ids:
                trees.append(tree)
                ids.add(key)
                n -= 1

    return trees




