import argparse
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

"""
Makes a chord (play-able with sounddevice) from a list of frequencies of the notes in the chord
"""
def make_chord(frequencies: list, samplerate: int = 44100, duration: float = 0.2):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    waves = [np.sin(2 * np.pi * f * t) for f in frequencies]
    chord = np.sum(waves, axis=0)
    chord *= 1 / np.max(np.abs(chord))

    fade_len = int(0.01 * samplerate)
    fade = np.linspace(1, 0, fade_len)
    chord[-fade_len:] *= fade

    return chord

"""
Visualises a Markov chain.
Plots the Markov chain states as a directed graph, with edge weights (for first order chains only) representing conditional probabilities of the transition from source chord to sink chord.
The vertex for `highlighted_chord` is shown in a different colour.
"""
def plot_graph_from_adj(adj: dict, previous_chords: list[str] = [], current_chord: str = None, mode: str = None):
    G = nx.DiGraph()
    for curr, targets in adj.items():
        for next, freq in targets.items():
            G.add_edge(curr[-1], next, weight=freq / sum(targets.values()))

    pos = nx.spiral_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    node_colors = [("black" if node==current_chord else ("teal" if node in previous_chords else "aqua")) for node in G.nodes()] if mode=="fit" else [("black" if node==current_chord else ("firebrick" if node in previous_chords else "lightcoral")) for node in G.nodes()]

    plt.clf()
    nx.draw(G, pos, node_color=node_colors, arrows=True)
    if len(list(adj.keys())[0])==1:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    plt.pause(0.3)

"""
FOR look_back=1:
Visualises training and returns a Markov chain whose states are the chords in the given file, which must contain a sequence of space-separated chords (chords are comma-separated integer frequencies).

The Markov chain is stored as a dict whose:
- key is a 1-tuple containing the previous chord as a string
- value is a dict of (next chord : conditional frequency) pairs

It learns the probability distribution of the next chord conditional on the previous chord.

FOR look_back>1:
Returns an n-gram model (n=look_back), which does not have the Markov property. No visualisation is performed.
"""
def fit_markov_chain(filename: str, look_back: int = 1, play: bool=False, plot: bool=False) -> dict:
    infile = open(filename, "r")
    text = infile.read()
    infile.close()

    chord_sequence = text.split()
    adj = {}
    iters = len(chord_sequence)-look_back
    for i in range(iters):
        if tuple(chord_sequence[i:i+look_back]) not in adj.keys():
            adj[tuple(chord_sequence[i:i+look_back])] = {}

        # update conditional frequency
        adj[tuple(chord_sequence[i:i+look_back])][chord_sequence[i+look_back]] = adj[tuple(chord_sequence[i:i+look_back])].get(chord_sequence[i+look_back], 0) + 1

        if play:
            sd.play(make_chord([int(x) for x in chord_sequence[i+look_back].split(',')], samplerate=44100, duration=0.2), samplerate=44100)
            if not plot:
                sd.wait()

        if plot:
            plot_graph_from_adj(adj, previous_chords=chord_sequence[i:i+look_back], current_chord=chord_sequence[i+look_back], mode="fit")

        print(f"[fitting {((i+1)*100)//iters}%] For chord(s) {tuple(chord_sequence[i:i+look_back])}: updated probability of transition to {chord_sequence[i+look_back]} to {(adj[tuple(chord_sequence[i:i+look_back])][chord_sequence[i+look_back]])/sum(adj[tuple(chord_sequence[i:i+look_back])].values()):.4f}")
        
    return adj

"""
[NOTE: The following description is for a Markov chain, but the function will also work for an n-gram model]
Given a fitted Markov chain as described above, composes and plays a new piece starting from a random chord.

For each 'current chord', it will sample a new chord from the conditional distribution of the next chord given the current chord, which is assumed to have already been learned.

Crashes if it encounters a dead-end, i.e., there is no possible new chord.
"""
def compose_piece(markov_chain: dict, piece_length: int = 1000, save_path: str = "output.wav", play: bool=True, plot: bool=True):
    audio_sequence = []
    prev_state = list(markov_chain.keys())[0]

    for i in range(piece_length):
        # simulate the conditional distribution
        current_chord = random.choices(list(markov_chain[prev_state].keys()), weights=list(markov_chain[prev_state].values()))[0]

        print(f"[generating {((i+1)*100)//piece_length}%] Transition from", prev_state, "to", current_chord)

        chord = make_chord([int(x) for x in current_chord.split(',')], samplerate=44100, duration=0.2)
        audio_sequence.append(chord)

        if play:
            sd.play(chord, samplerate=44100)
            if not plot:
                sd.wait()
        
        if plot:
            plot_graph_from_adj(markov_chain, previous_chords=list(prev_state), current_chord=current_chord, mode="simulate")

        prev_state = tuple((list(prev_state)[1:]) + [current_chord])

    wav_write(save_path, 44100, (np.concatenate(audio_sequence)*32767).astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a markov chain to a music piece, simulate it ans save the resulting composition in a .wav file")

    parser.add_argument("--input_path", type=str, help="A path to the piece to fit to, which should be a text file containing only space-separated chords. A chord is a comma-separated list of integer frequencies.")
    parser.add_argument("--look_back", type=int, default=1, help="The number of previous chords to condition the next chord on in the learned distribution.")
    parser.add_argument("--play", action="store_true", help="Enable playing the piece while fitting and while simulating?")
    parser.add_argument("--plot", action="store_true", help="Enable visualising the transition graph while fitting and while simulating?")
    parser.add_argument("--output_path", type=str, default="output.wav", help="The path to the .wav file to save the musical composition in.")
    parser.add_argument("--composition_length", type=int, default=3000, help="The length of the generated composition (in chords). 3000 chords is about 10 minutes.")
    
    args = parser.parse_args()
    
    crappy_composer = fit_markov_chain(args.input_path, look_back=args.look_back, play=args.play, plot=args.plot)
    compose_piece(crappy_composer, piece_length=args.composition_length, save_path=args.output_path, play=args.play, plot=args.plot)

