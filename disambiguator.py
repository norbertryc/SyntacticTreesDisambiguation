from data_preprocessing import *


def set_random_initial_best_choices(forest):

    forest = deepcopy(forest)

    # sprawdzenie poprawnosci lasu i ewentualne wypisanie komunikatu
    #_check_sentence(forest,"forest")

    nids = [int(x.attrib["nid"]) for x in forest.getroot().findall("node")]

    for nid in nids:
        children = forest.getroot().find(".//node[@nid='"+str(nid)+"']").findall("children")
        if len(children) == 1:
            children[0].set("current_best_choice","true")
        elif len(children) > 1:
            random_choice = np.random.choice(len(children))
            for c in range(len(children)):
                if c == random_choice:
                    children[c].set("current_best_choice","true")
                else:
                    children[c].set("current_best_choice","false")

    return forest


def get_current_best_tree(forest):
    
    """
 
    
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
    def add_current_best_children(current_node_old):
        
        current_node_new = ET.SubElement(root_new, current_node_old.tag, current_node_old.attrib)
        
        features = current_node_old.getchildren()
        # kazdy "node" jest terminalem albo nieterminalem i ma opis wlasnosci
        # i tutaj wyciagamy te wlasnosci z wezla innego niz "children"
        for feature in features:
            if feature.tag != "children": 
                current_node_new.append(feature)
        
        
        children_old = current_node_old.findall('children[@current_best_choice="true"]')
        # powinno byc tylko jedno takie dziecko
        
        assert len(children_old) <= 1, 'More than one children has current_best_choice="true"'
        
        if len(children_old) == 0: #jestesmy w lisciu wiec konczymy dzialanie funkcji
            return None
        
        #random_children_old = children_old[np.random.choice(len(children_old),1)[0]]
        children_new = ET.SubElement(current_node_new, children_old[0].tag, children_old[0].attrib)
        for child_old in children_old[0].getchildren():
            x = ET.SubElement(children_new, child_old.tag, child_old.attrib)
            next_node = root_old.find('.//node[@nid="' + x.attrib["nid"] + '"]')
            # jesli wpiszemy node'om znacznik to to wlaczyc: assert next_node.attrib["current_best_choice"] == "true"
            add_current_best_children(next_node)
        
    
    # wezel startowy (przyjmujemy, ze node z id=0 jest zawsze pierwszy):
    # TODO: upewnic sie czy to jest poprawne podejscie - czy moze byc inny wezel poczatkowym
    node_0 = root_old.find('.//node[@nid="0"][@chosen="true"]')  # current_best_choice
    
    # konstruujemy drzewo:
    add_current_best_children(node_0)
    
    current_best_tree = ET.ElementTree(root_new)

    ## Sprawdzenie poprawnosci drzewa
    #assert is_positive(positive_tree), """Something gone wrong - tree is not positive"""
        
        
    return current_best_tree
        
  

def get_disjunctive_nodes(forest):
    
    _check_sentence(forest)
    
    nids = [int(x.attrib["nid"]) for x in forest.getroot().findall("node")]
    
    disjunctive_nodes_ids = []
    
    for nid in nids:
        children = forest.getroot().find(".//node[@nid='"+str(nid)+"']").findall("children")
        if len(children) > 1:
            disjunctive_nodes_ids.append(nid)
            
    return disjunctive_nodes_ids

def get_trees_with_different_choice(forest, nid):
    
    forest = deepcopy(forest)
    
    trees = []
    
    children = forest.getroot().find(".//node[@nid='"+str(nid)+"']").findall("children")
    for i in range(len(children)):
        children[i].set("current_best_choice","false") 
    children[0].set("current_best_choice","true")  
    
    trees.append(get_current_best_tree(forest))
    
    
    for i in range(1,len(children)):
        children[i-1].set("current_best_choice","false") 
        children[i].set("current_best_choice","true")  
        trees.append(get_current_best_tree(forest))

    return trees



def which_best_tree(trees, model):
    
    probs = []
    for tree in trees:

        directory = "Data/tmp"
        write_dependency_format(transform_to_dependency_format(tree), directory, overwrite=True)

        

        data0 = load_stanford_data4(directory+"/labels.txt", directory+"/parents.txt",directory+"/tokens.txt",directory+"/rules.txt",w2vecs["words2ids"],True,s['batch_size'],s['nc'])

        data_rules = load_stanford_data4(directory+"/labels.txt", directory+"/parents.txt",directory+"/rules.txt",directory+"/rules.txt",rnn.rules2ids,True,s['batch_size'],s['nc'])
        data_rules = [x[0] for x in data_rules]

        data = [data0[i]+[data_rules[i]] for i in range(len(data0))]


        probs.append(model.predict_proba(data[0][0],data[0][4],data[0][1], data[0][3])[-1,1])

    best_choice = np.argmax(probs)
        
    return best_choice, probs[best_choice]

def update_current_best_choices(forest, nid, best_choice):
    
    
    #forest = deepcopy(forest)
    
    children = forest.getroot().find(".//node[@nid='"+str(nid)+"']").findall("children")
    
    for i in range(len(children)):
        if i == best_choice:
            children[i].set("current_best_choice","true")
        else:
            children[i].set("current_best_choice","false") 
    
    
    #return forest


def disambiguate(forest, model, n_init = "auto", greedy_search_treshold=100, early_stopping="auto", max_iter=1000):
    
    
    """
    
    ta funkcja musi tylko i wylacznie ujednoznaczniac - a nie liczyc statystyki...
    
    
    
    WPROWADZIC MODYFIKACJE, ZE JEST DRZEWO WYNIKOWE NIE MA PRZYNAJMNIEJ NP. 40% PSTWA BYCIA POPRAWNYM 
    TO POWTORZYC SZUKANIE NA INNEJ LOSOWEJ INICJALIZACJI. PRZYJAC OGRANICZENI, ZE SZUKAMY NP MAKS 5 RAZY
    (BO ZDAZA SIE ZE POPRAWNE MA MALO PROCENT)
    
    BO DZIEJE SIE TAK, ZE NA KILKASET LASOW TRAFI SIE KILKA, KTORE MAJA KILKASET TYSIECY DRZEW I SLABA JAKOSC NA TYCH TRZECH
    CIAGNIE GLOBALNE F1 DO BEZNADZIEJENEJ WARTOSCI
    
    DLA DYZYCH LASOW TRZEBA DAC DUO WIEKSZY EARLY_STOPPING, 
    BO TAKI MALY EWIDENTCIE NIE WYSTARCZA ZEBY ZDAZYC ZNALEZC SENSOWNE DRZEWO
    
    CZY MOZE ROZPATRYWAC WSZYSTKIE WEZLY NA RAZ, A NIE LOSOWO WYBRANE I ZACHLANNIE PODEJMOWAC DECYZJE?
    ALBO POSREDNIA OPCAJA JAK W DRZEWIE W SKLEARNIE - LOSOWAC KILKA WEZLOW I SPOSROD NICH SZUKAC
    
    
    
    """
    
    
    
    
    history = [], []
    positive_tree = get_positive_tree(forest) 
    
    number_of_trees = number_of_trees_in_forest(forest)
    
    if n_init == "auto":
        if number_of_trees < 10000:
            n_init=3
        elif number_of_trees < 500000:
            n_init = 3
        else:
            n_init = 1
            
    if early_stopping == "auto":
        if number_of_trees < 5000:
            early_stopping = 20
        else:
            early_stopping = 50
    
    if number_of_trees <= greedy_search_treshold:
        
        possible_tree_variants = get_all_trees_randomly(forest)
        best_choice, prob = which_best_tree(possible_tree_variants, model)
        #to do: zoptymalizowac - sprawdzac tylko opcje inne niz dotychczasowa i update'owac tylko jest znalala sie lepsza opcja
        #to do: zoptymalizowac tak, zeby zapisywac stany ukryte poddrzewa dla kazdego wezla zeby nie przeliczac od nowa wszystkieg
        chosen_tree = possible_tree_variants[best_choice]
        history[0].append(scores(positive_tree, chosen_tree)[2])
        history[1].append(prob)
            
        return None , history, None, None
    
    else:
        
        forest_with_best_tree = set_random_initial_best_choices(forest)
        best_prob = 0
        best_history = history
        
        for e in range(n_init):
            
            history = [], []
            current_forest = set_random_initial_best_choices(forest)

            disjunctive_nodes = set(get_disjunctive_nodes(current_forest))

            checked_nodes = set()

            for i in range(max_iter):

                nodes_left_to_check = disjunctive_nodes.difference(checked_nodes)

                if len(nodes_left_to_check) == 0:
                    break
                else:
                    random_disjunctive_node = random.sample(nodes_left_to_check,1)[0]

                #print(random_disjunctive_node)

                possible_tree_variants = get_trees_with_different_choice(current_forest, random_disjunctive_node)
                best_choice, prob = which_best_tree(possible_tree_variants, model)
            #to do: zoptymalizowac - sprawdzac tylko opcje inne niz dotychczasowa i update'owac tylko jest znalala sie lepsza opcja
                update_current_best_choices(current_forest, random_disjunctive_node, best_choice)

                chosen_tree = get_current_best_tree(current_forest)
                history[0].append(scores(positive_tree, chosen_tree)[2])
                history[1].append(prob)

                if i > 2 and history[1][-1] > history[1][-2]: # jezeli znalezlismy lepsze drzewo, to:
                    checked_nodes.clear() # czyscimy zestaw sprawdzonych  wierzcholkow
                    disjunctive_nodes = set(get_disjunctive_nodes(current_forest)) # uaktualniamy zestaw mozliwych wierzcholkow
                else:
                    checked_nodes.add(random_disjunctive_node) # dodajemy wierzolek do sprawdzonych, zeby drugi raz nie sprawdzac tego samego

                if prob > best_prob:
                    forest_with_best_tree = deepcopy(current_forest)
                    best_prob = prob
                    best_history = history
                    
                if i > early_stopping and history[1][-1] == history[1][-early_stopping]:
                    break
                
    
        #przypisujemy node'om atrybuty, ze zostaly wybrane lub nie do najlepszego drzewa:
    
    nodes_ids_chosen_in_best_tree = [0]+[int(x.attrib["nid"]) for children in forest.findall(".//children/[@current_best_choice='true']") for x in children.findall("child")]
    
    num_nodes = number_of_nodes(forest)
    
    for nid in range(num_nodes):
        if nid in nodes_ids_chosen_in_best_tree:
            forest.find(".//node[@nid='"+str(nid)+"']").set("current_best_choice","true")         
        else:
            forest.find(".//node[@nid='"+str(nid)+"']").set("current_best_choice","false") 
  
    nodes_ids_chosen_in_positive_tree = [0]+[int(x.attrib["nid"]) for x in forest.findall("node[@chosen='true']")]

    labels_best = np.zeros(num_nodes)
    labels_best[nodes_ids_chosen_in_best_tree] = 1

    labels_positive = np.zeros(num_nodes)
    labels_positive[nodes_ids_chosen_in_positive_tree] = 1
    
    
    return forest, history, labels_best, labels_positive
