import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_from_laplacian(
    laplacian,
    tickers=None,
    layout='spring',
    node_size=300,
    with_labels=True
):
    """
    Plot a graph constructed from a Laplacian matrix.

    Reconstructs the adjacency matrix from the provided Laplacian,
    builds a NetworkX graph, and plots it using the specified layout.
    Nodes can be optionally labeled with provided tickers.

    Parameters
    ----------
    laplacian : numpy.ndarray
        Square (n x n) Laplacian matrix (D - A).
    tickers : list of str, optional
        List of node labels of length n. If provided, must match
        the dimension of laplacian.
    layout : {'spring', 'circular', 'kamada_kawai'}, optional
        Graph layout. Default is 'spring'.
    node_size : int, optional
        Size of nodes in the plot. Default is 300.
    with_labels : bool, optional
        Whether to display node labels. Default is True.

    Raises
    ------
    ValueError
        If tickers length does not match graph size, or if layout
        string is not recognized.

    Returns
    -------
    None
    """
    # Reconstruct adjacency matrix
    degree_matrix = np.diag(np.diag(laplacian))
    adjacency_matrix = degree_matrix - laplacian

    # Build the graph
    graph = nx.from_numpy_array(adjacency_matrix)

    # Relabel nodes if tickers are provided
    if tickers is not None:
        if len(tickers) != graph.number_of_nodes():
            raise ValueError(
                f"Length of tickers ({len(tickers)}) does not match "
                f"graph size ({graph.number_of_nodes()})"
            )
        mapping = {i: tickers[i] for i in graph.nodes()}
        graph = nx.relabel_nodes(graph, mapping)

    # Select layout
    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError(f"Unknown layout '{layout}'")

    # Draw the graph
    plt.figure(figsize=(6, 6))
    nx.draw(
        graph,
        pos,
        node_size=node_size,
        with_labels=with_labels,
        edge_color='gray'
    )
    plt.title('Graph from Laplacian')
    plt.axis('off')
    plt.show()



def plot_regressor_coeffs(
    b,
    time=None,
    regressor_names=None,
    title='Regressor Coefficients Over Time'
):
    """
    Plot each regressor coefficient series across time steps.

    Parameters
    ----------
    b : numpy.ndarray
        Array of shape (J, M), where J is the number of coefficients
        and M is the number of time steps.
    time : array-like, optional
        Sequence of length M representing x-axis values. If None,
        uses 0..M-1.
    regressor_names : list of str, optional
        Names of the J regressors. If provided, must have length J.
    title : str, optional
        Title of the plot.

    Raises
    ------
    ValueError
        If regressor_names length does not match number of coefficients.

    Returns
    -------
    None
    """
    # Validate shapes
    J, M = b.shape

    # Validate regressor names
    if regressor_names is not None:
        if len(regressor_names) != J:
            raise ValueError(
                f"regressor_names must have length {J}, got {len(regressor_names)}"
            )
    else:
        regressor_names = [f'Coeff {j}' for j in range(J)]

    # Default time axis
    if time is None:
        time = np.arange(M)

    # Plot each coefficient series
    plt.figure()
    for j in range(J):
        plt.plot(time, b[j], label=regressor_names[j])

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
