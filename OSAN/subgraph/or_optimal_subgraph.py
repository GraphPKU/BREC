# see https://developers.google.com/optimization for the OR-Tools

from typing import List, Tuple, Optional

from ortools.linear_solver import pywraplp
import torch

from data.data_utils import edgeindex2neighbordict, get_ptr, get_connected_components

MAX_ITER_STEPS = 10


def get_basic_problem(value_list: List[List[int]], node_per_subgraphs)\
        -> Tuple[pywraplp.Solver, List[List[Optional[pywraplp.Variable]]]]:
    """

    :param value_list:
    :param node_per_subgraphs:
    :return:
    """
    n_subgraph, n_nodes = len(value_list), len(value_list[0])
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # variable
    x = [[None] * n_nodes for _ in range(n_subgraph)]
    for i in range(n_subgraph):
        for j in range(n_nodes):
            x[i][j] = solver.IntVar(0, 1, f'x[{i}][{j}]')

    # y = [[None] * graph.num_edges for _ in range(n_subgraph)]
    # for i in range(n_subgraph):
    #     for j in range(graph.num_edges):
    #         y[i][j] = solver.IntVar(0, 1, f'y[{i}][{j}]')

    # obj
    objective = solver.Objective()
    for i in range(n_subgraph):
        for j in range(n_nodes):
            objective.SetCoefficient(x[i][j], value_list[i][j])
    objective.SetMaximization()

    # coveredness
    if node_per_subgraphs * n_subgraph >= n_nodes:
        for j in range(n_nodes):
            solver.Add(sum([x[i][j] for i in range(n_subgraph)]) >= 1)

    # size of subgraphs
    for i in range(n_subgraph):
        solver.Add(sum([x[i][j] for j in range(n_nodes)]) == node_per_subgraphs)

    # # for each edge selected, its nodes are selected
    # for i in range(n_subgraph):
    #     for j in range(graph.num_edges):
    #         solver.Add(x[i][edge_index[j][0]] >= y[i][j])
    #         solver.Add(x[i][edge_index[j][1]] >= y[i][j])
    #
    # # two adjacent nodes selected, then their edge is selected
    # for i, (n1, n2) in enumerate(edge_index):
    #     for j in range(n_subgraph):
    #         solver.Add(y[j][i] <= x[j][n1])
    #         solver.Add(y[j][i] <= x[j][n2])
    #         solver.Add(y[j][i] >= x[j][n1] + x[j][n2] - 1)

    return solver, x


def get_solution_as_tensor(x: List[List[pywraplp.Variable]], device: torch.device = 'cpu'):
    """

    :param x:
    :param device:
    :return:
    """
    n_subgraph, n_nodes = len(x), len(x[0])
    solution = torch.empty(n_subgraph, n_nodes, dtype=torch.float32, device=device)
    for i in range(n_subgraph):
        for j in range(n_nodes):
            solution[i, j] = x[i][j].solution_value()
    return solution


def get_or_suboptim_subgraphs(value_tensor: torch.Tensor, node_per_subgraphs: int) -> torch.Tensor:
    """

    :param value_tensor:  shape (n_nodes, n_subgraphs)
    :param node_per_subgraphs:
    :return:
    """
    n_nodes, n_subgraph = value_tensor.shape

    if node_per_subgraphs >= n_nodes:
        return torch.ones_like(value_tensor, dtype=torch.float32, device=value_tensor.device)

    value_list = value_tensor.t().cpu().tolist()

    solver, x = get_basic_problem(value_list, node_per_subgraphs)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        pass
    else:
        raise ValueError("No solution")

    solution = get_solution_as_tensor(x, value_tensor.device)
    return solution.t()


def get_or_optim_subgraphs(graph, value_tensor, node_per_subgraphs, verbose=False):
    """

    :param graph:
    :param value_tensor:
    :param node_per_subgraphs:
    :param verbose:
    :return:
    """
    n_nodes, n_subgraph = value_tensor.shape

    if node_per_subgraphs >= n_nodes:
        return torch.ones_like(value_tensor, dtype=torch.float32, device=value_tensor.device)

    value_list = value_tensor.t().cpu().tolist()

    solver, x = get_basic_problem(value_list, node_per_subgraphs)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        pass
    else:
        raise ValueError("No solution")

    neighbor_dict = edgeindex2neighbordict(graph.edge_index.cpu().numpy(), graph.num_nodes)

    component_connectedness_constraint_idx = [[] for _ in range(n_subgraph)]
    optim_set = set()

    for _ in range(MAX_ITER_STEPS):
        if verbose:
            print(f'iter: {_}')
        solution = get_solution_as_tensor(x)

        row, col = torch.where(solution)
        split_idx = get_ptr(row)
        col = col.cpu().tolist()
        solution = solution.cpu().tolist()

        early_break = True
        for i in range(len(split_idx) - 1):
            if i in optim_set:
                if verbose:
                    print(f'{i} already optimal')
                continue

            if verbose:
                print(f'dealing with subgraph {i}')
            allnodes = col[split_idx[i]: split_idx[i + 1]]
            components = get_connected_components(allnodes, neighbor_dict)
            if len(components) == 1:
                while component_connectedness_constraint_idx[i]:
                    # remove those constraints and add the constraints of solution
                    constraint_idx_range = component_connectedness_constraint_idx[i].pop()
                    for constraint_idx in constraint_idx_range:
                        solver.constraints()[constraint_idx].Clear()
                for j in range(n_nodes):
                    solver.Add(x[i][j] == solution[i][j])
                optim_set.add(i)
            else:
                early_break = False
                num_constrain_before = solver.NumConstraints()

                if verbose:
                    print(f'components: {components}')
                for component in components:
                    adjacent_node_set = set()
                    for node in component:
                        for neighbor in neighbor_dict[node]:
                            if neighbor not in component:
                                adjacent_node_set.add(x[i][neighbor])
                    for node in component:
                        solver.Add(sum(adjacent_node_set) >= x[i][node])

                num_constrain_after = solver.NumConstraints()
                component_connectedness_constraint_idx[i].append(range(num_constrain_before, num_constrain_after))

        if verbose:
            print()

        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            pass
        else:
            raise ValueError("No solution")

        if early_break:
            break

    solution = get_solution_as_tensor(x, value_tensor.device)
    return solution.T
