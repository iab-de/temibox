import pytest
import matplotlib.pyplot as plt

from temibox.pipeline.dag import DAG


def test_dag_structure():

    d = DAG() \
        .add("a", 1) \
        .add("b", 1) \
        .add("a1", 1.1, depends_on=["a"]) \
        .add("a2", 1.2, depends_on=["a"]) \
        .add("c", 3, depends_on=["b", "a1"]) \
        .add("d", 4, depends_on=["a", "b"])

    assert d.children.keys() == {"a", "b"}, "root children do not match"

    assert d.children["a"].children.keys() == {"a1", "a2", "d"}, "node-a children do not match"
    assert d.children["a"].children["a1"].children.keys() == {"c"}, "node-a1 children do not match"
    assert d.children["a"].children["a2"].children.keys() == set(), "node-a2 children do not match"
    assert d.children["a"].children["d"].children.keys() == set(), "node-d children do not match"

    assert d.children["b"].children.keys() == {"c", "d"}, "node-b children do not match"
    assert d.children["b"].children["c"].children.keys() == set(), "node-c children do not match"
    assert d.children["b"].children["d"].children.keys() == set(), "node-d children do not match"

    assert len(d.parents) == 0, "root has no parents"
    assert d.name == "--root--", "root name is wrong"

    assert {p.name for p in d.children["a"].parents} == {"--root--"}, "node-a parents are wrong"
    assert {p.name for p in d.children["b"].parents} == {"--root--"}, "node-b parents are wrong"

    assert {p.name for p in d.children["a"].children["a1"].parents} == {"a"}, "node-a1 parents are wrong"
    assert {p.name for p in d.children["a"].children["a2"].parents} == {"a"}, "node-a2 parents are wrong"
    assert {p.name for p in d.children["b"].children["c"].parents} == {"b", "a1"}, "node-c parents are wrong"
    assert {p.name for p in d.children["b"].children["d"].parents} == {"a", "b"}, "node-c parents are wrong"

    # Root node
    assert all([d.root == c.root for c in d.children.values()]), "root for level 1 does not match"
    assert all([d.root == c.root for c in d.children["a"].children.values()]), "root for level 2 (a) does not match"
    assert all([d.root == c.root for c in d.children["b"].children.values()]), "root for level 2 (a) does not match"

    assert all([d.root == c.root for c in d.children.values()]), "root for level 1 does not match"
    assert all([d.root == c.root for c in d.children["a"].children.values()]), "root for level 2 (a) does not match"
    assert all([d.root == c.root for c in d.children["b"].children.values()]), "root for level 2 (a) does not match"

def test_dag_walk():

    d = DAG() \
        .add("a", 1) \
        .add("b", 1) \
        .add("a1", 1.1, depends_on=["a"]) \
        .add("a2", 1.2, depends_on=["a"]) \
        .add("c", 3, depends_on=["b", "a1"]) \
        .add("d", 4, depends_on=["a", "b"])

    expected = ["a", "b", "a1", "a2", "d", "c"]
    actual = []
    for name, value in d.walk():
        actual.append(name)

    assert all([actual[i] == expected[i] for i in range(len(expected))]), "Walk order does not match"


def test_dag_predicate():

    d = DAG() \
        .add("a", 1) \
        .add("b", 2) \
        .add("c", 3, depends_on=["a"]) \
        .add("d", 4, depends_on=["a"]) \
        .add("e", 5, depends_on=["b"]) \
        .add("f", 6, depends_on=["b"]) \
        .add("g", 7, depends_on=["a", "b"])

    predicate_even = lambda x: x % 2 == 0
    predicate_odd  = lambda x: x % 2 == 1

    expected_even = ["b", "d", "f"]
    expected_odd  = ["a", "c", "g", "e"]

    for predicate, expected in [(predicate_even, expected_even), (predicate_odd, expected_odd)]:
        actual = []
        for name, value in d.walk(predicate = predicate):
            actual.append(name)

        assert all([actual[i] == expected[i] for i in range(len(expected))]), "Walk order does not match for predicate"

def test_dag_dependencies():

    d = DAG() \
        .add("a", 1) \
        .add("b", 2) \
        .add("c", 3, depends_on=["a"]) \
        .add("d", 4, depends_on=["a"]) \
        .add("e", 5, depends_on=[]) \
        .add("f", 6, depends_on=None) \
        .add("g", 7)

    expected = {"a", "b", "e", "f", "g"}
    assert d.children.keys() == expected, "Root children do not match"

def test_dag_get():

    d = DAG() \
        .add("a", 1) \
        .add("b", 2) \
        .add("a1", 5, depends_on=["a"]) \
        .add("a2", 6, depends_on=["a"]) \
        .add("c", 3, depends_on=["b", "a1"]) \
        .add("d", 4, depends_on=["a", "b"])

    assert d.get("a").value  == 1, "Value of a does not match"
    assert d.get("b").value  == 2, "Value of b does not match"
    assert d.get("a1").value == 5, "Value of a1 does not match"
    assert d.get("a2").value == 6, "Value of a2 does not match"
    assert d.get("c").value  == 3, "Value of c does not match"
    assert d.get("d").value  == 4, "Value of d does not match"
    assert d.get("e") is None, "Value of e does not match"

def test_dag_detached_insert():

    d = DAG() \
        .add("a", 1) \
        .add("b", 2) \
        .add("c", 3)

    node = d.children["a"]

    node.add("d", 4, depends_on = ["a", "c"])

    assert d.nodes.keys() == {"a", "b", "c", "d"}, "Detached insert does not work"
    assert node.root == d.root, "Detached root does not match"
    assert d.children["a"].children.keys() == {"d"}, "Child insert failed (a)"
    assert d.children["c"].children.keys() == {"d"}, "Child insert failed (c)"
    assert d.children["b"].children.keys() == set(), "Child insert failed (b)"

    assert {p.name for p in d.children["a"].children["d"].parents} == {"a", "c"}, "Child parents do not match"


def test_dag_errors():

    # OK
    d = DAG().add("a", 1).add("b", 2).add("c", 3)
    assert d.nodes.keys() == {"a", "b", "c"}

    # Duplicate name
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add("a", 3)

    # Using _ROOTNAME as node name
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add(DAG._ROOTNAME, 3)

    # Nonexisting dependency 1
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add("c", 3, depends_on=["d"]).add("d", 4)

    # Nonexisting dependency 2
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add("c", 3, depends_on=["a", "d"]).add("d", 4)

    # Missing name 1
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add("", 3)

    # Missing name 2
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add(None, 3)

    # Missing value
    with pytest.raises(Exception):
        d = DAG().add("a", 1).add("b", 2).add("c", None)

def test_display():

    d = DAG() \
        .add("a", 1) \
        .add("b", 2) \
        .add("a1", 5, depends_on=["a"]) \
        .add("a2", 6, depends_on=["a"]) \
        .add("c", 3, depends_on=["b", "a1"]) \
        .add("d", 4, depends_on=["a", "b"])

    fig, ax = d.display(title = "example", predicates = [lambda x: True])

    assert isinstance(fig, plt.Figure), "Wrong type for fig"
    assert isinstance(ax, plt.Axes), "Wrong type for ax"

    plt.close()