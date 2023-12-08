#!/usr/bin/env python3

'''
This sample code serves as a toy example for the use of D-Wave's Ocean SDK.
For more complete documentation, please visit:
https://docs.ocean.dwavesys.com/en/stable/getting_started.html
'''

def run_light_switch_game(qpu_arch='pegasus', use_classical=False, 
        num_reads=10, show_inspector=False, plot_emb_path=None):
    '''
    Light switch game.

    Params:
        qpu_arch : QPU architecture to use. Either 'pegasus' (that is the newer
            architecture) or 'chimera' (the old one).
        use_classical : set to True to use D-Wave's classical Ising solver such 
            that you don't have to use up your D-Wave Leap minutes for testing.
        num_reads : count of samples to request from the D-Wave QPU or classical
            sampler. Don't use too high of a value on the QPU as that might 
            use up your Leap minutes very quickly.
        show_inspector : set to True to show problem inspector in the end.
        plot_emb_path : supply a string path to plot the embedding
    '''
    import matplotlib.pyplot as plt

    # general dwave dependencies
    import dwave
    import dwave.embedding
    import dwave.inspector
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.cloud import Client
    from dwave.cloud.exceptions import SolverNotFoundError
    import dimod
    from dimod.reference.samplers import ExactSolver
    from minorminer import find_embedding
    import neal

    # dependencies for plotting connectivity graph
    import networkx as nx
    import dwave_networkx as dnx

    # general math and Python dependencies
    import math
    import numpy as np
    import itertools

    # define self bias (h) and coupling strengths (J)

    # The one from D-Wave, but actually missing a couple of couplings so I've 
    # filled those J_{ij}s as 0
    #J = np.array(
    #        [[0, -1, 0.8, 0.1, 0, 0],
    #         [-1, 0, -0.3, 0, 0.2, 0],
    #         [0.8, -0.3, 0, 0, -0.7, 0.3],
    #         [0.1, 0, 0, 0, 0, 0],
    #         [0, 0.2, -0.7, 0, 0, 0],
    #         [0, 0, 0.3, 0, -0.7, 0]]
    #        )

    #h = - E_k * np.array([-1, 0])
    #J = - E_k * np.array(
    #        [[0, 1],
    #         [1, 0]])


    # Just two unbiased coupled switches
    # h = np.array([0.0, 0.0])
    # J = np.array(
    #     [[ 0.0,-1.0],
    #      [-1.0, 0.0]]
    # )

    # 5 switches example but just self bias
    # h = np.array([0.5, -0.2, 0.1, 1, -0.8])
    # J = np.array(
    #         [[0.0, 0.0, 0.0, 0.0, 0.0],
    #          [0.0, 0.0, 0.0, 0.0, 0.0],
    #          [0.0, 0.0, 0.0, 0.0, 0.0],
    #          [0.0, 0.0, 0.0, 0.0, 0.0],
    #          [0.0, 0.0, 0.0, 0.0, 0.0]]
    #         )


    # 5 switches example
    # h = np.array([0.5, -0.2, 0.1, 1, -0.8])
    # J = np.array(
    #        [[0.0, 0.9, 0.0, 0.0, 0.0],
    #         [0.9, 0.0, 0.4,-0.3, 0.0],
    #         [0.0, 0.4, 0.0, 0.0, 0.2],
    #         [0.0,-0.3, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.2, 0.0, 0.0]]
    #        )


    # 5 switches example with more edges
    h = np.array([0.5, -0.2, 0.1, 1, -0.8])
    J = np.array(
           [[0.0, 0.9, 0.0, 0.7, 0.0],
            [0.9, 0.0, 0.4,-0.3, 0.0],
            [0.0, 0.4, 0.0, 0.0, 0.2],
            [0.7,-0.3, 0.0, 0.0,-0.1],
            [0.0, 0.0, 0.2,-0.1, 0.0]]
           )


    N = len(h)

    # create edgelist (note that {} initializes Python dicts)
    linear = {}         # qubit self-bias
    quadratic = {}      # inter-qubit bias
    for i in range(N):
        linear[i] = h[i]
        for j in range(i+1, N):
            if J[i][j] != 0:
                quadratic[(i,j)] = J[i][j]

    # construct a bqm containing the provided self-biases (linear) and couplings
    # (quadratic). Specify the problem as SPIN (Ising).
    print('Constructing BQM...')
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.SPIN)

    # get DWave sampler and target mapping edgelist
    print('Choosing solver...')
    client = Client.from_config()
    solver = None
    try:
        if qpu_arch == 'zephyr':
            solver = client.get_solver('Advantage2_prototype1.1').id
        elif qpu_arch == 'pegasus':
            solver = client.get_solver('Advantage_system4.1').id
        elif qpu_arch == 'chimera':
            solver = client.get_solver('DW_2000Q_6').id
        else:
            raise ValueError('Specified QPU architecture is not supported.')
    except SolverNotFoundError:
        print(f'The pre-programmed D-Wave solver name for architecture '
                '\'{qpu_arch}\' is not available. Find the latest available '
                'solvers by:\n'
                'from dwave.cloud import Client\nclient = Client.from_config()\n'
                'client.get_solvers()\nAnd update this script.')
        raise
    # get the specified QPU
    dwave_sampler = DWaveSampler(solver=solver)

    # run the problem
    use_result = []
    sampler = None
    response = None
    if use_classical:
        print('Choosing classical sampler...')
        sampler = neal.SimulatedAnnealingSampler()
    else:
        print('Choosing D-Wave QPU as sampler...')
        sampler = EmbeddingComposite(dwave_sampler)
    response = sampler.sample(bqm, num_reads=50)
    print('Problem completed from selected sampler.')

    # plot the embedding if specified
    if not use_classical and plot_emb_path != None:
        print(f'Plotting embedding to {plot_emb_path}...')
        embedding = response.info['embedding_context']['embedding']
        plt.figure(figsize=(16,16))
        T_nodelist, T_edgelist, T_adjacency = dwave_sampler.structure
        if qpu_arch == 'pegasus':
            G = dnx.pegasus_graph(16,node_list=T_nodelist)
            dnx.draw_pegasus_embedding(G, embedding, node_size=8, cmap='rainbow')
        elif qpu_arch == 'chimera':
            G = dnx.chimera_graph(16,node_list=T_nodelist)
            dnx.draw_chimera_embedding(G, embedding, node_size=8, cmap='rainbow')
        # Note: I haven't looked into how to make dnx properly plot a Zephyr
        # embedding. If there's interest, shoot me (Sam) an email.
        #elif qpu_arch == 'zephyr':
        #    G = dnx.zephyr_graph(16,node_list=T_nodelist)
        #    dnx.draw_zephyr_embedding(G, embedding, node_size=8, cmap='rainbow')
        plt.savefig(plot_emb_path)

    # take first result from response
    use_result = [*response.first.sample.values()]
    # NOTE that response.record contains all returned results and some 
    # statistics. You can inspect what's inside by using pdb or reading the 
    # dimod.SampleSet documentation.

    import pdb; pdb.set_trace()

    # show dwave web inspector if specified
    if show_inspector and not use_classical:
        print('\nOpening problem inspector on your browser.')
        dwave.inspector.show(response)

if __name__ == '__main__':
    run_light_switch_game(qpu_arch='pegasus', use_classical=False, show_inspector=True,
            plot_emb_path='embedding.pdf')
