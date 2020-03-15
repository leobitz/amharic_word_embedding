# Amharic Word Representation Resources

This repository contains datatests, evaluations and helper scripts to manipulation the 

## Datasets and Evalutation sets
The data can be downloaded from [here](https://drive.google.com/file/d/1FTtDbWG5fmpsiAWS7w2lBm77-5jKWljH/view?usp=sharing). It is zip file that is hosted on the Google Drive. 

It contains the following files:

1. `charset.txt` The Fidel alphasyllabary, with rows representing onsets (consonants) in traditional order; columns represent nuclei (vowels).
2. `clean_corpus.txt` 16M-token corpus of Amharic text, after a "cleaning" pre-processing; one line per document.
3. `abj-am-clean_corpus.txt` As #2, but "Abjadized" -- namely, with all 6th Form (6th column in #1), corresponding to a "short i" or no vowel.
4. `alpha-clean_corpus.txt` As #2, but "Alphabetized" -- namely, with all 6th Form + a separate vowel symbol like "a" or "e" or "i" etc.
5. `analogy-evaluation.txt` An Amharic adaptation (1.7k analogies) of the Google word analogy task (Mikolov et al, 2013).
6. `abj-am-analogy.txt` As #5, but "Abjadized" -- namely, with all 6th Form (6th column in #1), corresponding to a "short i" or no vowel.
7. `alpha-am-analogy.txt` As #5, but "Alphabetized" -- namely, with all 6th Form + a separate vowel symbol like "a" or "e" or "i" etc.
8. `arabic-evaluation.txt` An Arabic adaptation (14k analogies) of the Google word analogy task (Zahran et al, 2015).

## Scripts

