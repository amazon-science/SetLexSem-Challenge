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


def make_sampler_name_from_hps(sampler_hps):
    """
    Create a formatted string name for a sampler based on its
    hyperparameters.
    """
    components = [
        f"MA-{sampler_hps['m_A']}",
        f"MB-{sampler_hps['m_B']}",
        f"L-{sampler_hps['item_len']}",
    ]
    if "n" in sampler_hps:
        n = None if sampler_hps["item_len"] else sampler_hps["n"]
        components.insert(0, f"N-{n}")

    if sampler_hps.get("overlap_fraction") is not None:
        components.append(f"O-{sampler_hps['overlap_fraction']}")

    if sampler_hps.get("decile_num") is not None:
        components.append(f"Decile-{sampler_hps['decile_num']}")

    return "_".join(components)


class Sampler:
    """
    Base class for samplers.

    Parameters
    ----------
    m_A : int
        Number of items to include in sampled set A.
    m_B : int
        Number of items to include in sampled set B.
    item_len : int, optional
        Length constraint for sampled items.
    random_state : Random, optional
        Random number generator.

    Raises
    ------
    ValueError
        If m is greater than n.
    """

    def __init__(self, m_A: int, m_B: int, item_len=None, random_state=None):
        self.m_A = m_A
        self.m_B = m_B
        self.item_len = item_len
        if isinstance(self.item_len, str):
            self.item_len = eval(self.item_len)
        if random_state is None:
            self.random_state = random.Random()
        else:
            self.random_state = random_state

    def __call__(self):
        """
        Sample two sets of items.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError()

    def __str__(self):
        """
        Create a string for the parameters of the generated data.

        Returns
        -------
        str
            Filename string.
        """
        return (
            f"{self.__class__.__name__} "
            f"({self.m_A=}, {self.m_B=}, {self.item_len=})"
        )

    def make_filename(self):
        """
        Create a string for the parameters of the generated data.

        Returns
        -------
        str
            Filename string.
        """

        return f"MA-{self.m_A}_MB-{self.m_B}_L-{self.item_len}"

    def create_sampler_for_k_shot(self):
        """
        Return the instance itself.

        Returns
        -------
        Sampler
            A reference to the instance itself.
        """
        return self

    def to_dict(self):
        """
        Convert sampler parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary containing sampler parameters.
        """
        return {
            "m_A": self.m_A,
            "m_B": self.m_B,
            "item_len": self.item_len,
            "set_type": self.get_members_type(),
            "decile_group": self.get_decile_group(),
            "overlap_fraction": self.get_overlap_fraction(),
        }

    def get_decile_group(self):
        return None

    def get_subset_size(self):
        return None

    def get_overlap_fraction(self):
        return None

    def get_members_type(self):
        return None


def filter_words(words, item_len):
    """
    Filter a list of words to only include those of a specific length.

    Parameters
    ----------
    words : list
        List of words to filter.
    item_len : int
        Desired length of words to keep.

    Returns
    -------
    list
        Filtered list of words with the specified length.

    Raises
    ------
    AssertionError
        If item_len is less than 1 or greater than the longest word.
    """
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
    Sampler for basic English words.

    Parameters
    ----------
    m_A : int
        Number of words to include in sampled set A.
    m_B : int
        Number of words to include in sampled set B.
    words : list or set of str, optional
        Custom word list to sample from.
    item_len : int, optional
        Length constraint for sampled words.
    pos : str, optional
        A WordNet part-of-speech tag. One of wn.ADJ (adjective), wn.ADJ_SAT
        (satellite adjective), wn.ADV (adverb), wn.NOUN, or wn.VERB.
    random_state : Random, optional
        Random number generator.
    """

    def __init__(
        self,
        m_A: int,
        m_B: int,
        words: Optional[Union[List[str], Set[str]]] = None,
        item_len=None,
        pos: Optional[str] = None,
        random_state=None,
    ):
        super().__init__(
            m_A, m_B, item_len=item_len, random_state=random_state
        )

        words = ENGLISH_WORDS if not words else words

        if pos:
            wn_parts_of_speech = {
                wn.ADJ,
                wn.ADJ_SAT,
                wn.ADV,
                wn.NOUN,
                wn.VERB,
            }
            if pos not in wn_parts_of_speech:
                raise ValueError(
                    f"'pos' must be one of {wn_parts_of_speech}, not '{pos}'."
                )
            LOGGER.debug(
                f"Dictionary size before filtering by {pos} {len(words)}."
            )
            # Reduce dictionary to those lemmata with at least one synset
            # having the given part of speech.
            words = [word for word in words if len(wn.synsets(word, pos=pos))]
            LOGGER.debug(
                f"Dictionary size after filtering by {pos} {len(words)}."
            )

        if self.item_len is None:
            self.possible_options = words
        else:
            assert self.item_len >= 1, "item_len should be greater than 0"
            self.possible_options = filter_words(words, self.item_len)

    def __call__(self):
        """
        Sample two sets of words.

        Returns
        -------
        tuple of set
            Two sets of sampled words.
        """

        A = set(self.random_state.sample(self.possible_options, self.m_A))
        B = set(self.random_state.sample(self.possible_options, self.m_B))
        return A, B

    def get_members_type(self):
        """
        Get the type of members in the sampled sets.

        Returns
        -------
        str
            Type of members ("words").
        """
        return "words"


class BasicNumberSampler(Sampler):
    """
    Sampler for numbers.

    Parameters
    ----------
    n : int
        Upper bound of the range to sample from.
    m : int
        Number of numbers to include in each sampled set.
    item_len : int, optional
        Length constraint for sampled numbers.
    random_state : Random, optional
        Random number generator.
    """

    def __init__(
        self,
        n: int,
        m_A: int,
        m_B: int,
        item_len=None,
        random_state=None,
    ):
        super().__init__(
            m_A, m_B, item_len=item_len, random_state=random_state
        )
        self.n = None if self.item_len else n

        if self.n is not None:
            if m_A > n:
                raise ValueError(
                    f"m ({m_A}) should be greater than n ({n}) but {m_A} <= {n}"
                )
            if m_B > n:
                raise ValueError(
                    f"m ({m_B}) should be greater than n ({n}) but {m_B} <= {n}"
                )

        self.init_range_filter()

    def init_range_filter(self):
        """
        Initialize the range of possible numbers based on constraints.
        """
        if self.item_len is None:
            self.possible_options = range(0, self.n)
        else:
            assert self.item_len >= 1, "item_len should be greater than 0"
            range_f = 10 ** (self.item_len - 1)
            range_l = 10 ** (self.item_len)
            self.possible_options = range(range_f, range_l)

    def __call__(self):
        """
        Sample two sets of numbers.

        Returns
        -------
        tuple of set
            Two sets of sampled numbers.
        """
        A = set(self.random_state.sample(self.possible_options, self.m_A))
        B = set(self.random_state.sample(self.possible_options, self.m_B))
        return A, B

    def get_members_type(self):
        """
        Get the type of members in the sampled sets.

        Returns
        -------
        str
            Type of members ("numbers").
        """
        return "numbers"

    def make_filename(self):
        """
        Create a string for the parameters of the generated data.

        Returns
        -------
        str
            Filename string.
        """
        return f"N-{self.n}_MA-{self.m_A}_MB-{self.m_B}_L-{self.item_len}"

    def to_dict(self):
        """
        Convert sampler parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary containing sampler parameters.
        """
        return super().to_dict().update({"n": self.n})


class OverlapSampler(Sampler):
    """
    Sampler that creates overlapping sets.

    Parameters
    ----------
    sampler : Sampler
        Base sampler to use for initial sampling.
    overlap_fraction : float, optional
        Fraction of overlap between sets.
        overlap_fraction % of the elements in the smaller set
        must also be in the larger set.
    overlap_n : int, optional
        Number of overlapping items.
    """

    def __init__(
        self,
        sampler: Sampler,
        overlap_fraction: int = None,
        overlap_n: int = None,
    ):
        super().__init__(
            sampler.m_A,
            sampler.m_B,
            item_len=sampler.item_len,
            random_state=sampler.random_state,
        )
        self.sampler = sampler
        self.overlap_fraction = overlap_fraction

        A_init, B_init = self.sampler()

        m_small, m_large = (
            (self.m_A, self.m_B)
            if self.m_A < self.m_B
            else (self.m_B, self.m_A)
        )

        if self.overlap_fraction is not None:
            assert (
                self.overlap_fraction <= 1 and self.overlap_fraction >= 0
            ), f"overlap fraction ({self.overlap_fraction}) has to be 0<X<1"
            self.overlap_n = int(m_small * self.overlap_fraction)

            if self.overlap_n == 0:
                LOGGER.warning(
                    f"{overlap_fraction=} is too small (n={self.overlap_n})"
                )
        else:
            self.overlap_n = overlap_n

        self.nonoverlap_n = max(
            0,
            (
                m_large - self.overlap_n
                if self.m_A <= self.m_B
                else m_large - self.overlap_n - self.m_B
            ),
        )

    def __call__(self):
        """
        Sample two sets with specified overlap.

        Returns
        -------
        tuple of set
            Two sets with specified overlap.

        Raises
        ------
        StopIteration
            If unable to create sets with specified overlap after 100 attempts.
        """
        A, B = self.sampler()

        counter = 0
        while len(A.intersection(B)) != self.overlap_n or len(B) != self.m_B:
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
        """
        Get the type of members in the sampled sets.

        Returns
        -------
        str
            Type of members (includes "overlapping" prefix).
        """
        return f"overlapping_{self.sampler.__class__.__name__}"

    def make_filename(self):
        """
        Create a string for the parameters of the generated data.

        Returns
        -------
        str
            Filename string including overlap information.
        """
        name_pre = self.sampler.make_filename()
        return f"{name_pre}_O-{self.overlap_n}"

    def get_overlap_fraction(self):
        return self.overlap_fraction


def get_clean_hyponyms(
    random_state,
    save_json=0,
    filename=os.path.join(PATH_DATA_ROOT, "hyponyms.json"),
):
    """
    Get a list of clean hyponyms and optionally save them to a JSON file.

    Parameters
    ----------
    random_state : RandomState
        Random state for shuffling.
    save_json : int, optional
        Whether to save the result to JSON. Default is 0.
    filename : str, optional
        Path to save the JSON file. Default is "hyponyms.json" in
        PATH_DATA_ROOT.

    Returns
    -------
    list
        List of clean hyponym sets.
    """
    hyperhypo = list(find_hypernyms_and_hyponyms())
    clean_hyponyms = postprocess_hyponym_sets(hyperhypo, random_state)

    if save_json:
        with open(filename, "w") as f:
            json.dump(clean_hyponyms, f)

    return clean_hyponyms


def postprocess_hyponym_sets(hyperhypo, random_state):
    """
    Process hyponym sets to clean and simplify the lexical forms.

    Converts sets of hyponyms to strings. The hyponyms are WordNet Synsets.
    A synset can have multiple lexical forms (obtained via
    `Synset.lemma_names()`). Some lemma names are simple variations of each
    other. We aggressively filter them out.

    Parameters
    ----------
    hyperhypo : list
        List of hypernym-hyponym pairs.
    random_state : RandomState
        Random state for shuffling.

    Returns
    -------
    list
        Cleaned list of hyponym sets.
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
    Sampler for creating sets of words that may be confusing to language models.

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

    Parameters
    ----------
    m_A : int
        Number of words to include in sampled set A.
    m_B : int
        Number of words to include in sampled set B.
    item_len : int, optional
        Length constraint for sampled words (not supported).
    random_state : Random, optional
        Random number generator.
    with_replacement : bool, optional
        Whether to sample with replacement.
    swap_set_elements : bool, optional
        Whether to swap elements between sets.
    swap_n : int, optional
        Number of elements to swap between sets.
    random_state_mix_sets : Random, optional
        Random number generator for mixing sets.

    Raises
    ------
    ValueError
        If m is greater than 30.
    """

    def __init__(
        self,
        m_A: int,
        m_B: int,
        item_len=None,
        random_state=None,
        with_replacement=False,
        swap_set_elements=False,
        swap_n: int = None,
        random_state_mix_sets=None,
    ):
        super().__init__(
            m_A, m_B, item_len=item_len, random_state=random_state
        )
        if self.item_len is not None:
            warnings.warn(
                "DeceptiveWordSampler does not support `item_len` argument",
                category=UserWarning,
            )
        if self.m_A > 30 or self.m_B > 30:
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
        max_set_size = max(self.m_A, self.m_B)
        f = partial(by_length, min_length=max_set_size)
        # filtered hyponyms
        self.possible_options = list(filter(f, self.clean_hyponyms))

    def __call__(self):
        """
        Sample two sets of words with potential mixing.

        Returns
        -------
        tuple of set
            Two sets of sampled words.
        """
        if not self.with_replacement:
            # When we're not using replacement, the selected set of words is
            # removed from the set of options. So when we're not using
            # replacement, make a defensive copy, so we don't end up with an
            # empty list of options.
            possible_options = list(self.possible_options)
        else:
            possible_options = self.possible_options
        A = self.choose_hyponyms(possible_options, self.m_A)
        B = self.choose_hyponyms(possible_options, self.m_B)
        if self.swap_set_elements:
            A, B = self.mix_sets(A, B, subset_size=self.swap_n)
        return A, B

    def load_hyponym_sets(self, filename):
        """
        Load hyponym sets from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file containing hyponym sets.

        Returns
        -------
        list
            List of hyponym sets.
        """
        with open(filename) as f:
            hyponyms = json.load(f)
        return hyponyms

    def choose_hyponyms(
        self, hyponyms, set_size, with_replacement=False, normalize=True
    ):
        """
        Choose a set of hyponyms and return a random subset.

        Parameters
        ----------
        hyponyms : list
            List of hyponym sets to choose from.
        with_replacement : bool, optional
            Whether to sample with replacement.
        normalize : bool, optional
            Whether to normalize the chosen hyponyms.

        Returns
        -------
        set
            Set of chosen hyponyms.
        """
        hyponym_list = list(self.random_state.choice(hyponyms))
        if not with_replacement:
            hyponyms.remove(hyponym_list)
        self.random_state.shuffle(hyponym_list)
        prepared = hyponym_list[:set_size]
        prepared = set(prepared)
        return prepared

    def mix_sets(self, A, B, subset_size=None):
        """
        Mix elements between two sets.

        Choose a particular subset size for mixing and swap a subset of that
        size between A and B. A and B are already shuffled, so we just take the
        first elements.

        Parameters
        ----------
        A : set
            First set of words.
        B : set
            Second set of words.
        subset_size : int, optional
            Size of the subset to mix between sets.

        Returns
        -------
        tuple of set
            Two sets after mixing elements.

        Raises
        ------
        ValueError
            If subset_size is larger than either set.
        """
        smaller_set_size = min((len(A), len(B)))
        if not subset_size:
            subset_size = self.random_state_mix_sets.randint(
                1, smaller_set_size
            )
        if subset_size > smaller_set_size:
            raise ValueError(
                f"Subset to mix ({subset_size}) is bigger than "
                f"either A ({len(A)}) or B ({len(B)})"
            )
        self.subset_size = subset_size

        A, B = list(A), list(B)
        a = A[-subset_size:]
        b = B[-subset_size:]
        A = A[:-subset_size] + b  # noqa: E203
        B = B[:-subset_size] + a  # noqa: E203
        return set(A), set(B)

    def get_members_type(self):
        """
        Get the type of members in the sampled sets.

        Returns
        -------
        str
            Type of members ("deceptive_words").
        """
        return "deceptive_words"

    def get_subset_size(self):
        return self.subset_size

    def make_filename(self):
        """
        Create a string for the parameters of the generated data.

        Returns
        -------
        str
            Filename string including decile information.
        """

        name_pre = f"MA-{self.m_A}_MB-{self.m_B}_L-{self.item_len}"

        if self.swap_set_elements:
            return f"{name_pre}_DeceptiveWords_Swapped-{self.subset_size}"
        else:
            return f"{name_pre}_DeceptiveWords"


class DecileWordSampler(BasicWordSampler):
    """
    Sampler for words from a specific frequency decile.

    Parameters
    ----------
    m : int
        Number of words to include in each sampled set.
    decile_num : int
        Decile number to sample from (1-10).
    item_len : int, optional
        Length constraint for sampled words.
    random_state : Random, optional
        Random number generator.
    """

    def __init__(
        self,
        m_A: int,
        m_B: int,
        decile_num: int,
        item_len=None,
        random_state=None,
    ):
        self.decile_num = decile_num
        self.deciles = self.load_deciles()[str(self.decile_num)]
        super().__init__(
            m_A,
            m_B,
            words=self.deciles,
            item_len=item_len,
            random_state=random_state,
        )

    def load_deciles(self):
        """
        Load word frequency deciles from a JSON file.

        Returns
        -------
        dict
            Dictionary of word frequency deciles.
        """
        with open(f"{PATH_DATA_ROOT}/deciles.json", "rt") as fh:
            deciles = json.load(fh)
        return deciles

    def get_members_type(self):
        """
        Get the type of members in the sampled sets.

        Returns
        -------
        str
            Type of members ("decile_words").
        """
        return "decile_words"

    def make_filename(self):
        """
        Create a string for the parameters of the generated data.

        Returns
        -------
        str
            Filename string including decile information.
        """
        name_pre = f"MA-{self.m_A}_MB-{self.m_B}_L-{self.item_len}"
        return f"{name_pre}_Decile-{self.decile_num}"

    def __str__(self):
        return (
            f"{self.__class__.__name__} "
            f"({self.m_A=}, {self.m_B=}, {self.item_len=}, {self.decile_num=})"
        )

    def get_decile_group(self):
        return self.decile_num


def normalize_lemma_name(lemma_name):
    """
    Replace underscores in lemma names with spaces.

    Parameters
    ----------
    lemma_name : str
        The lemma name to normalize.

    Returns
    -------
    str
        Normalized lemma name.
    """
    return lemma_name.replace("_", " ")


def is_lemma_simple(lemma):
    """
    Check if a lemma is simple (single token without complex characters).

    Parameters
    ----------
    lemma : str
        The lemma to check.

    Returns
    -------
    bool
        True if the lemma is simple, False otherwise.
    """
    for complex_character in "_-":
        if complex_character in lemma:
            return False
    return True


def contains_uppercase(synset):
    """
    Check if any lemma in the synset contains uppercase characters.

    Parameters
    ----------
    synset : Synset
        The synset to check.

    Returns
    -------
    bool
        True if any lemma contains uppercase, False otherwise.
    """
    for lemma_name in synset.lemma_names():
        if lemma_name.lower() != lemma_name:
            return True
    return False


def contains_character(synset, characters="-"):
    """
    Check if any lemma in the synset contains specified characters.

    Parameters
    ----------
    synset : Synset
        The synset to check.
    characters : str, optional
        Characters to look for. Default is "-".

    Returns
    -------
    bool
        True if any lemma contains specified characters, False otherwise.
    """
    for lemma_name in synset.lemma_names():
        for character in characters:
            if character in lemma_name:
                return True
    return False


def remove_substring_lemmata(lemma_names):
    """
    Remove any lemma that is a substring of another lemma.

    Parameters
    ----------
    lemma_names : list
        List of lemma names.

    Returns
    -------
    list
        Filtered list of lemma names without substrings.
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
    Create a queue of lemma pairs sorted by edit distance.

    Make a queue with edit distances as keys and lists of lemmata pairs as
    values. The elements are sorted, in ascending order, with the list of
    lemmata pairs with the least edit distance first.

    Parameters
    ----------
    lemma_names : list
        List of lemma names.

    Returns
    -------
    list
        Sorted queue of (edit_distance, lemma_pairs) tuples.
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
    Remove similar lemmata until the minimum pairwise edit distance is met.

    Parameters
    ----------
    lemma_names : list
        List of lemma names.
    random_state : RandomState
        Random state for shuffling.
    min_distance : int, optional
        Minimum edit distance. Default is 3.
    max_iteration : int, optional
        Maximum number of iterations. Default is 4.

    Returns
    -------
    list
        Filtered list of lemma names.

    Raises
    ------
    StopIteration
        If max_iteration is exceeded.
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

    Parameters
    ----------
    synset : Synset
        The synset to get hyponyms for.

    Returns
    -------
    set
        Set of all hyponyms.
    """
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())


def find_hypernyms_and_hyponyms():
    """
    Find hypernym-hyponym pairs, along with their distance.

    Yields
    ------
    tuple
        (hypernym, hyponyms) pairs.
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
    """
    Get the lengths of hyponym sets for each hypernym.

    Parameters
    ----------
    hyperhypo : list
        List of hypernym-hyponym pairs.

    Returns
    -------
    list
        List of hyponym set lengths.
    """
    lengths = [len(hh) for hh in hyperhypo]
    return lengths


def by_length(s, min_length=None, max_length=30):
    """
    Check if the length of a set is within specified bounds.

    We want the user to be able to choose a size for their sets. We also want
    to be careful that they're not too big. Sets of hyponyms with many elements
    are super generic (e.g. the hyponyms of the hypernym "entity") and aren't
    useful for our task.

    Parameters
    ----------
    s : set or list
        The set to check.
    min_length : int, optional
        Minimum length. Must be positive.
    max_length : int, optional
        Maximum length. Default is 30.

    Returns
    -------
    bool
        True if length is within bounds, False otherwise.

    Raises
    ------
    ValueError
        If min_length is not positive.

    >>> from functools import partial
    >>> f = partial(by_length, min_length=5)
    >>> hyperhypo = find_hypernyms_and_hyponyms()
    >>> filtered = list(filter(f, hyperhypo))
    """
    if not min_length:
        raise ValueError(f"min_length must be positive, not {min_length}")
    return min_length <= len(s) <= max_length
