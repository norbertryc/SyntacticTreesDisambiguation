from data_preprocessing import terminals


def scores(positive_tree, selected_tree):
    """
    Ta funkcja powinna byc w osobnym module pt "narzedzia do ewaluacji modeli.
    
    wszystkie inne do przetwarzania drzew powinny byc w module pt "ptrzetwarzanie danych"
    
    trzeci modul to narzedzia zwiazane bezposrednio z disambiguatorem
    """
    gold_nodes = set([node.attrib["nid"] for node in positive_tree.findall('node[@chosen="true"]')])
    selected_nodes = set([node.attrib["nid"] for node in selected_tree.findall('node')])
    
    # terminale zawsze wchodza do kazdego drzewa, wiec usuwamy je zeby liczyc miary tylko na nieterminalach
    terminal_nodes = set(terminals(positive_tree)[1])
    gold_nodes.difference_update(terminal_nodes)
    selected_nodes.difference_update(terminal_nodes)
    
    intersection = set.intersection(gold_nodes, selected_nodes)
    
    precision = len(intersection)/len(selected_nodes)
    recall = len(intersection)/len(gold_nodes)
    f1 = 2/(1/precision+1/recall)
    return precision, recall, f1
