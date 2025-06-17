# Usage
* Clone this repository
```sh
git clone https://github.com/ronitkunk/markovleafrag.git
```
* Install dependencies
```sh
pip install -r requirements.txt
```
* Create a text file containing the score for the piece you want to train the Markov chain on, as a sequence with space-separated chords. Chords are comma-separated unordered sets of integer frequencies.
```
77,155 103,207 415 155,207,261,311,622 415
```
* * there are sample files provdided like `joplin/mapleleafrag.txt` and `vivaldi/fourseasonsspring.txt`.
* Run `markov_composer.py` and follow the instructions in the terminal.
```python
python markov_composer.py
```
For example, while training a Markov chain on the Maple Leaf Rag sample score provided, the terminal would look something like this:
```
Please specify a path to the training piece, which should be a text file containing only space-separated chords. A chord is a comma-separated list of integer frequencies.
> joplin/mapleleafrag.txt
[training 0%] For chord(s) ('77,155',): updated probability of transition to 103,207 to 1.0000
[training 0%] For chord(s) ('103,207',): updated probability of transition to 415 to 1.0000
[training 0%] ...
```
The training piece will play while training, after which the generated piece will play.

# Thanks
The .txt scores for Joplin and Vivaldi compositions have been derived from CC-licensed MIDI files downloaded from [The Mutopia Project](https://www.mutopiaproject.org/ftp/).