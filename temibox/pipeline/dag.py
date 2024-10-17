import networkx as nx
import matplotlib.pyplot as plt
from typing import Type, TypeVar, Generator, Any, Callable, Generic

from ..traits import PipelineStep, Predictable, Evaluating, Supervising


T = TypeVar("T")
Node = TypeVar("Node", bound=Type["DAG(Generic[T])"])

class DAG(Generic[T]):
    r"""
    Simple implementation of a directed acyclic graph for use in a temibox pipeline
    """

    _ROOTNAME = "--root--"

    def __init__(self,
                 name: str | None = None,
                 value: T = None,
                 parents: list[Node] | None = None):

        self._name  = name
        self._value = value

        self._parents:  list[Node]      = parents or []
        self._children: dict[str, Node] = {}
        self._nodes:    dict[str, Node] = parents[0].nodes if parents else {}
        self._root:     Node            = parents[0].root if parents else self

    def _cname(self, name: str) -> str:
        return name.lower().strip()

    def add(self, name: str, value: Any, depends_on: list[str] | None = None) -> 'DAG':
        r"""
        Add a node to the DAG

        :param name: name of the node
        :param value: value of the node
        :param depends_on: list of names of dependencies

        :return: self (for method chaining)
        """

        if name is None or name == "" or value is None:
            raise Exception("Child node must have a name and a value")

        if (cname := self._cname(name)) in self._nodes:
            raise Exception(f"Entry with the name '{name}' already exists")

        if cname == self._ROOTNAME:
            raise Exception(f"Invalid name '-root-'")

        if not depends_on:
            self._nodes[cname]    = DAG(name, value, parents=[self])
            self._children[cname] = self._nodes[cname]
        else:
            self._nodes[cname] = DAG(name, value)
            for pname in [self._cname(d) for d in depends_on]:
                if pname not in self._nodes:
                    raise Exception(f"Unknown dependency '{pname}'")

                self._nodes[pname].children[cname] = self._nodes[cname]
                self._nodes[cname].parents.append(self._nodes[pname])
                self._nodes[cname]._root  = self._root
                self._nodes[cname]._nodes = self._nodes

        return self

    def get(self, name: str) -> Node | None:
        r"""
        Returns a node from the DAG based on the name

        :param name: name of the node

        :return: DAG-node, if it exists
        """

        return self._nodes.get(self._cname(name), None)

    def set_value(self, value: T):
        r"""
        Sets node's value

        :param value: new value of the node

        :return: None
        """

        self._value = value

    def get_dependencies(self, name) -> list[str]:
        r"""
        Return node's dependencies

        :param name: name of the node

        :return: list of names of dependencies
        """

        if node := self.get(name):
            return node.dependencies

        return []

    @property
    def root(self) -> Node:
        r"""
        Returns the root of the DAG

        :return: root node
        """

        return self._root

    # Node methods
    @property
    def dependencies(self) -> set[str]:
        r"""
        Returns own dependencies

        :return: set of dependency names
        """

        return {self._cname(p.name) for p in self.parents}

    @property
    def parents(self) -> list[Node]:
        r"""
        Returns list of parent nodes

        :return: list of nodes
        """

        return self._parents

    @property
    def children(self) -> dict[str, Node]:
        r"""
        Returns dictionary of children

        :return: dictionary of children
        """

        return self._children

    @property
    def nodes(self) -> dict[str, Node]:
        r"""
        Returns dictionary with all the nodes

        :return: dictionary of nodes
        """

        return self._nodes

    @property
    def values(self) -> list[T]:
        r"""
        Returns list of node values

        :return: list of values of type T
        """

        return [n.value for n in self.nodes.values()]

    @property
    def name(self) -> str:
        r"""
        Returns node's name

        :return: returns name
        """

        return self._name or self._ROOTNAME

    @property
    def value(self) -> Any:
        r"""
        Returns node's value

        :return: node's value
        """
        return self._value

    def walk(self,
             predicate: Callable[[T], bool] | None = None) -> Generator[tuple[str, T], None, None]:

        r"""
        Walks along the DAG based on the dependencies.

        An optional predicate can be used to filter the returned nodes.
        The predicate does not influence the walk path, only the returned
        nodes

        :param predicate: method accepting a value of type T and returning a bool

        :return: returns a generator
        """

        return DAGWalker(self.root, predicate).walk()

    def _get_dep_groups(self, dep_groups, predicate):

        for name, node in self.walk(predicate=predicate):
            if name not in dep_groups:
                dep_groups[name] = []

            for dep in self.get_dependencies(name):

                if dep not in dep_groups:
                    continue

                dep_groups[name].append(dep)

    def display(self, title: str, predicates: list[Callable[[PipelineStep], bool]]):
        r"""
        Draws the DAG as a network graph

        :param title: title of the graph
        :param predicates: list of predicates used to filter the DAG

        :return: plot figure and axis objects
        """

        dep_groups = {}
        for predicate in predicates:
            self._get_dep_groups(dep_groups, predicate)

        dep_groups_inverse = {}
        for name, ndeps in dep_groups.items():
            for dep in ndeps:
                if dep not in dep_groups_inverse:
                    dep_groups_inverse[dep] = [name]
                else:
                    dep_groups_inverse[dep].append(name)

        deps = []
        for name, dep in dep_groups_inverse.items():
            for d in dep:
                deps.append((name, d))

        graph = nx.DiGraph(deps)

        color_map = []
        for name in graph.nodes():
            node = self.get(name)

            if node is None:
                color_map.append('#B1DDF1')

            elif isinstance(node.value, Predictable):
                color_map.append('#084C61')

            elif isinstance(node.value, Evaluating):
                color_map.append('#DB3A34')

            elif isinstance(node.value, Supervising):
                color_map.append('#FFC857')

            else:
                color_map.append("#ECC8AF")

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.title(title)

        pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")
        labels = {}
        for i, label in enumerate(dep_groups.keys()):
            labels[label] = f"{i + 1}: {label}"

        nx.draw_networkx(
            graph,
            pos=pos,
            labels=labels,
            arrowsize=12,
            with_labels=True,
            node_size=1200,
            node_color=color_map,
            linewidths=1.0,
            width=1.0,
            font_size=6,
            edge_color="tab:gray"
        )

        return fig, ax


class DAGWalker(Generic[T]):
    r"""
    Generator used to walk the DAG
    """

    def __init__(self, root: Node, predicate: Callable[[T], bool] | None = None):
        self._root = root
        self._predicate = predicate
        self._visited: set[str] = set()

    def walk(self) -> Generator[tuple[str, T], None, None]:
        r"""
        Walks the DAG, yielding relevant nodes

        :return: relevant nodes
        """
        available: dict[str, Node] = self._root.children

        while True:

            # Yield all available
            for key, node in available.items():
                self._visited.add(key)

                if self._predicate and not self._predicate(node.value):
                    continue

                yield node.name, node.value

            # Identify next batch
            available_next = {}
            for key, node in available.items():
                if key not in self._visited:
                    continue

                for ckey, cnode in node.children.items():
                    if (cnode.dependencies & self._visited) == cnode.dependencies:
                        available_next[ckey] = cnode

            # continue or break
            if len(available_next):
                available = available_next
            else:
               return