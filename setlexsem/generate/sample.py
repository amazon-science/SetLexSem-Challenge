"""
Samplers of set operands, such as sets of numbers, of words, or of sets of
related words.

    >>> ignored = 1000
    >>> m = 10
    >>> from setlexsem.samplers import DeceptiveWordSampler
    >>> sampler = DeceptiveWordSampler(n=ignored, m=m)
    >>> sampler()
    (['alcoholism',
      'mania',
      'logorrhea',
      'phaneromania',
      'dipsomania',
      'agromania',
      'workhouse',
      'slammer',
      'jailhouse'],
     ['bastille',
      'brig',
      'poky',
      'camp',
      'borstal',
      'gulag',
      'compulsion',
      'trichotillomania',
      'onomatomania'])

"""

import json
import logging
import os
import random
import warnings

# FIXME make sampler for semantic collections of words.
from collections import defaultdict
from functools import partial
from operator import itemgetter
from typing import List, Optional, Set, Union

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import words

from setlexsem.constants import PATH_DATA_ROOT

ENGLISH_WORDS = list(set(w.lower() for w in words.words()))

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Discarded redundant search for Synset",
)


class Sampler:
    def __init__(self, n: int, m: int, item_len=None, random_state=None):
        if m > n:
            raise ValueError(
                f"m ({m}) should be greater than n ({n}) but {m} <= {n}"
            )
        self.n = n
        self.m = m
        self.item_len = item_len
        if random_state is None:
            self.random_state = random.Random()
        else:
            self.random_state = random_state

    def __call__(self):
        raise NotImplementedError()

    def __str__(self):
        return (
            f"{self.__class__.__name__} "
            f"({self.n=}, {self.m=}, {self.item_len=})"
        )

    def make_filename(self):
        """Create a string for the parameters of the generated data"""
        if self.item_len:
            n = None
        else:
            n = self.n

        return f"N-{n}_M-{self.m}_L-{self.item_len}"

    def create_sampler_for_k_shot(self):
        """Create a copy of the class"""
        return self


def make_sampler_name_from_hps(sampler_hps):
    # idx 0: SET_TYPES
    # idx 1: N
    # idx 2: M
    # idx 3: ITEM_LEN
    # idx 4: DECILE_NUM
    if sampler_hps[3]:  # if item length is set, we change N to None
        n = None
    else:
        n = sampler_hps[1]

    if "decile" in sampler_hps[0]:
        txt_out = "_".join(
            [
                f"N-{n}",
                f"M-{sampler_hps[2]}",
                f"L-{sampler_hps[3]}",
                f"Decile-{sampler_hps[4]}",
            ]
        )
    else:
        if len(sampler_hps) == 5:  # means we have overlap
            if sampler_hps[3]:
                if sampler_hps[4] is not None:
                    txt_out = "_".join(
                        [
                            f"N-{n}",
                            f"M-{sampler_hps[2]}",
                            f"L-{sampler_hps[3]}",
                            f"O-{sampler_hps[4]}",
                        ]
                    )
                else:
                    txt_out = "_".join(
                        [
                            f"N-{n}",
                            f"M-{sampler_hps[2]}",
                            f"L-{sampler_hps[3]}",
                        ]
                    )
            else:
                txt_out = "_".join(
                    [
                        f"N-{n}",
                        f"M-{sampler_hps[2]}",
                        f"L-{sampler_hps[3]}",
                    ]
                )
        else:
            txt_out = "_".join(
                [
                    f"N-{n}",
                    f"M-{sampler_hps[2]}",
                    f"L-{sampler_hps[3]}",
                ]
            )

    return txt_out


def filter_words(words, item_len):
    """returns a list of words that have length N"""
    assert item_len >= 1, "N should be greater than 0"
    assert item_len <= len(max(words, key=len)), (
        f"item_len (={item_len}) should be less than "
        f"the length of the longest word ({max(words, key=len)})"
    )

    new_word_list = [word for word in words if len(word) == item_len]

    assert all(
        len(word) == item_len for word in new_word_list
    ), f"words must have len={item_len}: {new_word_list}"

    return new_word_list


class BasicWordSampler(Sampler):
    """
    Return two sets of m (m=number of items) words sampled from n words in the
    English word list.

    If item_len is not None, the length of each word is limited to it.
    """

    def __init__(
        self,
        n: int,
        m: int,
        words: Optional[Union[List[str], Set[str]]] = None,
        item_len=None,
        random_state=None,
        **kwargs,
    ):
        super().__init__(n, m, item_len=item_len, random_state=random_state)

        words = ENGLISH_WORDS if not words else words

        if self.item_len is None:
            if self.n > len(words):
                # need to make the number of items smaller to enable sampling
                self.n = len(words)
            self.possible_options = self.random_state.sample(words, self.n)
        else:
            assert self.item_len >= 1, "item_len should be greater than 0"
            filtered_english_words = filter_words(words, self.item_len)
            if self.n > len(filtered_english_words):
                # need to make the number of items smaller to enable sampling
                self.n = len(filtered_english_words)
            self.possible_options = self.random_state.sample(
                filtered_english_words, self.n
            )

    def __call__(self):
        A = set(self.random_state.sample(self.possible_options, self.m))
        B = set(self.random_state.sample(self.possible_options, self.m))
        return A, B

    def get_members_type(self):
        return "words"


class BasicNumberSampler(Sampler):
    """
    Return two sets of m numbers sampled from 0 to n-1.

    If item_len is not None, the length of each number is limited, and n is
    overridden.
    """

    def __init__(
        self, n: int, m: int, item_len=None, random_state=None, **kwargs
    ):
        super().__init__(n, m, item_len=item_len, random_state=random_state)
        self.init_range_filter()

    def init_range_filter(self):
        if self.item_len is None:
            self.possible_options = range(0, self.n)
        else:
            assert self.item_len >= 1, "item_len should be greater than 0"
            range_f = 10 ** (self.item_len - 1)
            range_l = 10 ** (self.item_len)
            self.possible_options = range(range_f, range_l)

    def __call__(self):
        A = set(self.random_state.sample(self.possible_options, self.m))
        B = set(self.random_state.sample(self.possible_options, self.m))
        return A, B

    def get_members_type(self):
        return "numbers"


class OverlapSampler(Sampler):
    """
    Return two sets of m numbers that are overlapped by overlap_percentage
    """

    def __init__(
        self,
        sampler: Sampler,
        overlap_fraction: int = None,
        overlap_n: int = None,
    ):
        super().__init__(
            sampler.n,
            sampler.m,
            item_len=sampler.item_len,
            random_state=sampler.random_state,
        )
        self.sampler = sampler
        self.overlap_fraction = overlap_fraction

        A_init, _ = self.sampler()
        m = len(A_init)
        if self.overlap_fraction is not None:
            assert (
                self.overlap_fraction <= 1 and self.overlap_fraction >= 0
            ), f"overlap fraction ({self.overlap_fraction}) has to be 0<X<1"
            self.overlap_n = int(m * self.overlap_fraction)

            if self.overlap_n == 0:
                LOGGER.warning(
                    f"{overlap_fraction=} is too small (n={self.overlap_n})"
                )
        else:
            self.overlap_n = overlap_n

        self.nonoverlap_n = m - self.overlap_n

    def __call__(self):
        A, B = self.sampler()

        counter = 0
        while len(A.intersection(B)) != self.overlap_n or len(A) != len(B):
            A, B1 = self.sampler()
            A2, B2 = self.sampler()

            B = set(self.random_state.sample(list(A), self.overlap_n))
            B = B.union(
                set(
                    self.random_state.sample(
                        list(A2.union(B1, B2)), self.nonoverlap_n
                    )
                )
            )

            # raise error for while loop
            if counter > 100:
                raise StopIteration(
                    "Not enough possible options to make non-overlapping sets."
                    " Reduce the constraints or increase overlap fraction|n."
                )
            counter += 1

        return A, B

    def get_members_type(self):
        return f"overlapping_{self.sampler.__class__.__name__}"

    def make_filename(self):
        """Create a string for the parameters of the generated data"""
        name_pre = self.sampler.make_filename()
        return f"{name_pre}_O-{self.overlap_n}"


def get_clean_hyponyms(
    random_state,
    save_json=0,
    filename=os.path.join(PATH_DATA_ROOT, "hyponyms.json"),
):
    """
    Return a list of lists of hyponyms and save them
    """
    hyperhypo = list(find_hypernyms_and_hyponyms())
    clean_hyponyms = postprocess_hyponym_sets(hyperhypo, random_state)

    if save_json:
        with open(filename, "w") as f:
            json.dump(clean_hyponyms, f)

    return clean_hyponyms


def postprocess_hyponym_sets(hyperhypo, random_state):
    """
    Convert sets of hyponyms to strings. The hyponyms are WordNet Synsets.
    A synset can have multiple lexical forms (obtained via
    `Synset.lemma_names()`). Some lemma names are simple variations of each
    other. We aggressively filter them out.
    """
    clean_hyponyms = []
    for hyper, hypolist in hyperhypo:
        clean_hyponyms.append([])
        for hypo in hypolist:
            # hypo is one synset of a hyponym associated with a particular
            # hypernym
            lemma_names = hypo.lemma_names()
            # Remove lemmata with small edit distances between one another.
            try:
                lemma_names = remove_similar_lemmata(
                    lemma_names, random_state
                )
                simple_lemma_names = list(
                    filter(is_lemma_simple, lemma_names)
                )
                if len(simple_lemma_names):
                    clean_hyponyms[-1].extend(simple_lemma_names)
            except StopIteration:
                pass
    return clean_hyponyms


class DeceptiveWordSampler(Sampler):
    """
    Return two sets of m (m=number of items) words sampled from the words in
    the WordNet dictionary. The sets are constructed in a way that may be
    confusing to a language model. Here's how:

        1. The user asks for two sets of words of size m from this class.
        2. The class finds two separate sets of related words, A and B. All
           of the words in A are hyponyms (subtypes) of the same hypernym.
           All of the words in B are hyponyms of the same hypernym (which
           should be different from the hypernym of the elements of A, although
           I don't have a check to guard against that!).
        3. A subset of a random size s (1 <= s < m) is selected. Then s
           elements from A are moved to B and s elements from B are moved to A.
           This creates an artificial bifurcation within each of A and B.

    We will determine experimentally whether this bifurcation confuses language
    models when they attempt to perform set operations.

    If item_len is not None, the length of each word is limited to it.
    """

    def __init__(
        self,
        n: int,
        m: int,
        item_len=None,
        random_state=None,
        with_replacement=False,
        swap_set_elements=False,
        swap_n: int = None,
        random_state_mix_sets=None,
        **kwargs,
    ):
        super().__init__(n, m, item_len=item_len, random_state=random_state)
        if self.item_len is not None:
            warnings.warn(
                "DeceptiveWordSampler does not support `item_len` argument",
                category=UserWarning,
            )
        if self.m > 30:
            raise ValueError(
                "DeceptiveWordSampler won't sample sets larger than 30"
            )
        self.random_state_mix_sets = random_state_mix_sets
        self.with_replacement = with_replacement
        self.swap_set_elements = swap_set_elements
        self.swap_n = swap_n
        # hyperhypo = list(find_hypernyms_and_hyponyms())
        self.clean_hyponyms = self.load_hyponym_sets(
            os.path.join(PATH_DATA_ROOT, "hyponyms.json")
        )
        # self.postprocess_hyponym_sets(hyperhypo)
        f = partial(by_length, min_length=self.m)
        # filtered hyponyms
        self.possible_options = list(filter(f, self.clean_hyponyms))

    def __call__(self):
        if not self.with_replacement:
            # When we're not using replacement, the selected set of words is
            # removed from the set of options. So when we're not using
            # replacement, make a defensive copy, so we don't end up with an
            # empty list of options.
            possible_options = list(self.possible_options)
        else:
            possible_options = self.possible_options
        A = self.choose_hyponyms(possible_options)
        B = self.choose_hyponyms(possible_options)
        if self.swap_set_elements:
            A, B = self.mix_sets(A, B, subset_size=self.swap_n)
        return A, B

    def load_hyponym_sets(self, filename):
        with open(filename) as f:
            hyponyms = json.load(f)
        return hyponyms

    def choose_hyponyms(
        self, hyponyms, with_replacement=False, normalize=True
    ):
        """
        Choose a particular set of hyponyms and return a random subset of m of
        its elements.
        """
        hyponym_list = list(self.random_state.choice(hyponyms))
        if not with_replacement:
            hyponyms.remove(hyponym_list)
        self.random_state.shuffle(hyponym_list)
        prepared = hyponym_list[: self.m]
        prepared = set(prepared)
        return prepared

    def mix_sets(self, A, B, subset_size=None):
        """
        Choose a particular subset size for mixing and swap a subset of that
        size between A and B. A and B are already shuffled, so we just take the
        first elements.
        """
        if not subset_size:
            subset_size = self.random_state_mix_sets.randint(1, self.m)
        if subset_size > min((len(A), len(B))):
            raise ValueError(
                f"Subset to mix ({subset_size}) is bigger than "
                f"either A ({len(A)}) or B ({len(B)})"
            )
        A, B = list(A), list(B)
        a = A[-subset_size:]
        b = B[-subset_size:]
        A = A[:-subset_size] + b  # noqa: E203
        B = B[:-subset_size] + a  # noqa: E203
        return set(A), set(B)

    def get_members_type(self):
        return f"deceptive_words_{self.swap_set_elements}"


class DecileWordSampler(BasicWordSampler):
    """
    Return two sets of m (m=number of items) words sampled from n words in the
    English word list.

    If item_len is not None, the length of each word is limited to it.
    """

    def __init__(
        self,
        n: int,
        m: int,
        decile_num: int,
        item_len=None,
        random_state=None,
        **kwargs,
    ):
        self.decile_num = decile_num
        self.deciles = self.load_deciles()[str(self.decile_num)]
        super().__init__(
            n,
            m,
            words=self.deciles,
            item_len=item_len,
            random_state=random_state,
        )

    def load_deciles(self):
        with open(f"{PATH_DATA_ROOT}/deciles.json", "rt") as fh:
            deciles = json.load(fh)
        return deciles

    def get_members_type(self):
        return "decile_words"

    def make_filename(self):
        """Create a string for the parameters of the generated data"""
        if self.item_len:
            n = None
        else:
            n = self.n

        name_pre = f"N-{n}_M-{self.m}_L-{self.item_len}"
        return f"{name_pre}_Decile-{self.decile_num}"

    def __str__(self):
        return (
            f"{self.__class__.__name__} "
            f"({self.n=}, {self.m=}, {self.item_len=}, {self.decile_num=})"
        )


def normalize_lemma_name(lemma_name):
    """Replace underscores in lemma names with space."""
    return lemma_name.replace("_", " ")


def is_lemma_simple(lemma):
    """A lemma is simple when it's a single token."""
    for complex_character in "_-":
        if complex_character in lemma:
            return False
    return True


def contains_uppercase(synset):
    """Returns true if any lemma in the synset is uppercase (proper name?)."""
    for lemma_name in synset.lemma_names():
        if lemma_name.lower() != lemma_name:
            return True
    return False


def contains_character(synset, characters="-"):
    """
    Returns true if any lemma in the synset contains any of the characters.
    """
    for lemma_name in synset.lemma_names():
        for character in characters:
            if character in lemma_name:
                return True
    return False


def remove_substring_lemmata(lemma_names):
    """
    Remove any lemma that is a substring of another lemma.
    """
    substring_lemmata = set()
    # Ensure uniqueness and defensively copy.
    lemma_names = list(set(lemma_names))
    # Sort by length, so lemma 2 is never a substring of lemma 1.
    lemma_names.sort(key=len)
    for i, lemma_name1 in enumerate(lemma_names):
        for j, lemma_name2 in enumerate(lemma_names[i + 1 :]):  # noqa: E203
            if lemma_name1 in lemma_name2:
                substring_lemmata.add(lemma_name1)
    lemma_names_without_substrings = [
        ln for ln in lemma_names if ln not in substring_lemmata
    ]
    return lemma_names_without_substrings


def make_edit_distance_queue(lemma_names):
    """
    Make a queue with edit distances as keys and lists of lemmata pairs as
    values. The elements are sorted, in ascending order, with the list of
    lemmata pairs with the least edit distance first.
    """
    distances = defaultdict(list)
    for i, lemma_name1 in enumerate(lemma_names):
        for j, lemma_name2 in enumerate(lemma_names[i + 1 :]):  # noqa: E203
            distance = nltk.edit_distance(lemma_name1, lemma_name2)
            distances[distance].append([lemma_name1, lemma_name2])
    queue = sorted(distances.items(), key=itemgetter(0))
    return queue


def remove_similar_lemmata(
    lemma_names, random_state, min_distance=3, max_iteration=4
):
    """
    Remove a lemma until the mininum pairwise edit distance is greater than
    or equal to `min_distance`.
    """
    lemma_names = list(lemma_names)
    lemma_names = remove_substring_lemmata(lemma_names)
    queue = make_edit_distance_queue(lemma_names)
    iteration = 0
    while len(queue) and (queue[0][0] < min_distance):
        if iteration > max_iteration:
            raise StopIteration()

        # Remove one lemma at random from the least edit-distance pairs.
        # A random lemmata pair.
        lemmata_pair = random_state.choice(queue[0][1])
        # A random lemma from the pair.
        lemma_to_remove = random_state.choice(lemmata_pair)
        lemma_names.remove(lemma_to_remove)

        queue = make_edit_distance_queue(lemma_names)

        iteration += 1

    return lemma_names


def get_hyponyms(synset):
    """
    Get all the hyponyms of this synset.

    See https://stackoverflow.com/a/33662890 for origin of this code.
    """
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())


def find_hypernyms_and_hyponyms():
    """
    Find hypernym-hyponym pairs, along with their distance.

    Return generator of (hypernym, hyponym, distance) tuples.
    """
    for synset in wn.all_synsets():
        # Find all the hyponyms of this synset.
        if f"{synset}" not in [
            "Synset('restrain.v.01')",
            "Synset('inhibit.v.04')",
        ]:
            hyponyms = get_hyponyms(synset)
        # else:
        #     warnings.warn(
        #         f"Recursion error getting hyponyms of {synset}", UserWarning
        #     )

        if len(hyponyms):
            yield synset, hyponyms


def get_hyponym_set_lengths(hyperhypo):
    """Get the lengths of the sets of hyponyms of each hypernym."""
    lengths = [len(hh) for hh in hyperhypo]
    return lengths


def by_length(s, min_length=None, max_length=30):
    """
    We want the user to be able to choose a size for their sets. We also want
    to be careful that they're not too big. Sets of hyponyms with many elements
    are super generic (e.g. the hyponyms of the hypernym "entity") and aren't
    useful for our task.

    >>> from functools import partial
    >>> f = partial(by_length, min_length=5)
    >>> hyperhypo = find_hypernyms_and_hyponyms()
    >>> filtered = list(filter(f, hyperhypo))
    """
    if not min_length:
        raise ValueError(f"min_length must be positive, not {min_length}")
    return min_length <= len(s) <= max_length
