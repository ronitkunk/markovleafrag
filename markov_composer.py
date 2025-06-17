import random
import numpy as np
import sounddevice as sd

"""
Makes a chord (play-able with sounddevice) from a list of frequencies of the notes in the chord
"""
def make_chord(frequencies: list, samplerate: int, duration: float):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    waves = [np.sin(2 * np.pi * f * t) for f in frequencies]
    chord = np.sum(waves, axis=0)
    chord *= 1 / np.max(np.abs(chord))

    fade_len = int(0.01 * samplerate)
    fade = np.linspace(1, 0, fade_len)
    chord[-fade_len:] *= fade

    return chord

"""
FOR look_back=1:
Returns a Markov chain whose states are the chords in the given file, which must containi a sequence of space-separated chords (chords are comma-separated integer frequencies).

The Markov chain is stored as a dict whose:
- key is a 1-tuple containing the previous chord as a string
- value is a dict of (next chord : conditional frequency) pairs

It learns the probability distribution of the next chord conditional on the previous chord.

FOR look_back>1:
Returns an n-gram model (n=look_back), which does not have the Markov property
"""
def train_markov_chain(train_filename: str, look_back: int = 1) -> dict:
    infile = open(train_filename, "r")
    text = infile.read()
    infile.close()

    chord_sequence = text.split()
    adj = {}
    iters = len(chord_sequence)-look_back
    for i in range(iters):
        if tuple(chord_sequence[i:i+look_back]) not in adj.keys():
            adj[tuple(chord_sequence[i:i+look_back])] = {}

        if chord_sequence[i+look_back] not in adj[tuple(chord_sequence[i:i+look_back])].keys():
            adj[tuple(chord_sequence[i:i+look_back])][chord_sequence[i+look_back]] = 1
        else:
            adj[tuple(chord_sequence[i:i+look_back])][chord_sequence[i+look_back]] += 1
        
        print(f"[training {((i+1)*100)//iters}%] For chord(s) {tuple(chord_sequence[i:i+look_back])}: updated probability of transition to {chord_sequence[i+look_back]} to {(adj[tuple(chord_sequence[i:i+look_back])][chord_sequence[i+look_back]])/sum(adj[tuple(chord_sequence[i:i+look_back])].values()):.4f}")

        sd.play(make_chord([int(x) for x in chord_sequence[i+look_back].split(',')], samplerate=44100, duration=0.2), samplerate=44100)
        sd.wait()

    return adj

"""
[NOTE: The following description is for a Markov chain, but the function will also work for an n-gram model]
Given a trained Markov chain as described above, composes and plays a new piece starting from a random chord.

For each 'current chord', it will sample a new chord from the conditional distribution of the next chord given the current chord, which is assumed to have already been learned.

Crashes if it encounters a dead-end, i.e., there is no possible new chord.
"""
def play_piece(markov_chain: dict, piece_length: int = 1000):
    prev_state = random.choice(list(markov_chain.keys()))

    for i in range(piece_length):
        current_chord = random.choices(list(markov_chain[prev_state].keys()), weights=list(markov_chain[prev_state].values()))[0]

        print(f"[generating {((i+1)*100)//piece_length}%] Transition from", prev_state, "to", current_chord)

        sd.play(make_chord([int(x) for x in current_chord.split(',')], samplerate=44100, duration=0.2), samplerate=44100)
        sd.wait()

        prev_state = tuple((list(prev_state)[1:])+[current_chord])

if __name__ == "__main__":
    training_piece = input("Please specify a path to the training piece, which should be a text file containing only space-separated chords. A chord is a comma-separated list of integer frequencies.\n> ")
    
    crappy_music_composer = train_markov_chain(training_piece)
    play_piece(crappy_music_composer, piece_length=1000)