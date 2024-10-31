# coding: utf-8

import random
from itertools import product

from nltk.corpus import words

from setlexsem.generate.sample import Sampler

ENGLISH_WORDS = list(set(w.lower() for w in words.words()))


# define prompt config class
class PromptConfig:
    """This class maintains the config of the prompt"""

    def __init__(
        self,
        k_shot: int,
        type: str,
        approach: str,
        sampler: Sampler,
        operation: str = "None",
        is_fixed_shots: bool = True,
    ):
        self.k_shot = k_shot
        self.type = type
        self.approach = approach
        self.sampler = sampler
        self.operation = operation
        self.item_type = self.sampler.get_members_type()
        self.is_fixed_shots = is_fixed_shots

    def __str__(self):
        return (
            f"{self.__class__.__name__} (operation={self.operation},"
            f" k={self.k_shot}, type={self.type},"
            f" approach={self.approach}, item_type={self.item_type},"
            f" is_fixed_shots={self.is_fixed_shots})"
        )

    def get_instruction(self):
        return make_instruction_generator(self.type)(self.operation)

    def _define_kshots(self, sampler):
        """Given the sampler, we create a list of k-shot examples"""
        if self.k_shot == 0:
            return []

        examples = []
        for _ in range(self.k_shot):
            examples.append(sampler())

        return examples

    def getKShot(self):
        if self.is_fixed_shots:
            return self.getFixedKShot()
        else:
            return self.getDynamicKShot()

    def getFixedKShot(self):
        """Create K-shot examples that are fixed across examples"""
        new_sampler = self.sampler
        new_sampler.random_state = random.Random(13121)
        return self._define_kshots(new_sampler)

    def getDynamicKShot(self):
        """Create K-shot examples that are dynamic across examples"""
        new_sampler = self.sampler
        return self._define_kshots(new_sampler)

    # return as dictionary
    def to_dict(self):
        return {
            "k_shot": self.k_shot,
            "type": self.type,
            "approach": self.approach,
            "operation": self.operation,
            "item_type": self.item_type,
            "is_fixed_shots": self.is_fixed_shots,
            **self.sampler.to_dict(),
        }


# operations available
OPS = {
    "union",
    "intersection",
    "difference",
    "symmetric difference",
    "cartesian product",
}

# operation instructions in formal language
OP_INSTRUCTION_FORMAL_LANGUAGE = {
    "union": "Print the set union of A and B as a Python set.",
    "intersection": "Print the set intersection of A and B as a Python set.",
    "difference": "Print the set difference of A and B as a Python set.",
    "symmetric difference": "Print the symmetric set difference of A and B.",
    "cartesian product": (
        "Print the Cartesian product of A and B as as a Python list of "
        "tuples, all on one line."
    ),
}

# operation instructions in plain English
OP_INSTRUCTION_PLAIN_LANGUAGE = {
    "union": (
        "Print the set of members belonging to either A or B as "
        "a Python set."
    ),
    "intersection": (
        "Print the set of members belonging to both A and B as a Python set."
    ),
    "difference": (
        "Print the set of members belonging to A and not to B as "
        "a Python set."
    ),
    "symmetric difference": (
        "Print the set of members belonging to A or B but not both A and B "
        "as a Python set."
    ),
    "cartesian product": (
        'Print each pair of elements (a, b) where "a" is from A and "b" '
        "is from B as a Python list of tuples, all on one line."
    ),
}

OP_INSTRUCTION_PYTHONIC_CONCISENESS = {
    "union": "Take the elements that are in either set A or set B, and present them all together as a single set in Python.",
    "intersection": "Take the elements that are common to both set A and set B, and present them as a single set in Python.",
    "difference": "Take the elements that are in set A but not in set B, and present them as a single set in Python.",
    "symmetric difference": "Take the elements that are in either set A or set B, but not in both, and present them as a single set in Python.",
    "cartesian product": "Combine each element from set A with each element from set B, and present the resulting list of tuples in Python.",
}

OP_INSTRUCTION_FUNCTIONAL_STYLE = {
    "union": "Take the items from set A and the items from set B, combine them, and then display the full set of all those elements in Python.",
    "intersection": "Take the elements that are in both set A and set B, combine them into a single set, and display that set in Python.",
    "difference": "Take the elements that are in set A but not in set B, combine them into a single set, and display that set in Python.",
    "symmetric difference": "Take the elements that are in either set A or set B, but not both, combine them into a single set, and display that set in Python.",
    "cartesian product": "Combine each element from set A with each element from set B, and present the resulting list of tuples in Python.",
}

OP_INSTRUCTION_ITERATIVE_ACCUMULATION = {
    "union": "Start with an empty set, go through the elements in set A and add them, then go through the elements in set B and add them as well, and finally show the complete set containing all those elements in Python.",
    "intersection": "Start with an empty set, go through the elements in set A, and only add them to the set if they are also in set B, then show the final set.",
    "difference": "Start with an empty set, go through the elements in set A, and only add them to the set if they are not in set B, then show the final set.",
    "symmetric difference": "Start with an empty set, go through the elements in set A and add them if they are not in set B, then go through the elements in set B and add them if they are not in set A, and finally show the complete set.",
    "cartesian product": "Create an empty list, then go through each element in set A and each element in set B, creating a tuple with the two elements and adding it to the list, and finally display the complete list of tuples.",
}

# operation ground truth calculation
OP_GROUND_TRUTH = {
    "union": lambda a, b: a.union(b),
    "intersection": lambda a, b: a.intersection(b),
    "difference": lambda a, b: a.difference(b),
    "symmetric difference": lambda a, b: a.symmetric_difference(b),
    "cartesian product": lambda a, b: list(product(a, b)),
}


PROMPT_STOP = "Stop after printing."
PROMPT_DO_NOT_USE_CODE = "Do not write a code or script or use any tools."
PROMPT_EMPTY_SET = "The answer can be an empty set."
PROMPT_THINKING = (
    "Explain your step-by-step reasoning process in detail within "
    "<thinking></thinking> XML tags."
)
PROMPT_FINAL_ANSWER = (
    "At last, provide only the final answer as a mathematical set, without "
    "any code or additional context. Do not include anything other than your "
    "final answer in your response within <answer></answer> XML tags."
)


# templates for different approaches of prompt engineering
PROMPT_TEMPLATES = {
    # No prompt engineering; the LLM is instructed to return only the answer
    "baseline": " ".join(
        [
            "\nDo not explain your reasoning.",
            PROMPT_DO_NOT_USE_CODE,
            PROMPT_FINAL_ANSWER,
            PROMPT_STOP,
        ]
    ),
    # Allow empty set response in baseline mode
    "baseline_allow_empty": " ".join(
        [
            "\nDo not explain your reasoning.",
            PROMPT_DO_NOT_USE_CODE,
            PROMPT_FINAL_ANSWER,
            PROMPT_EMPTY_SET,
            PROMPT_STOP,
        ]
    ),
    # The system prompt specifies that the LLM is an expert in the domain
    "domain_expertise": " ".join(
        [
            "\nYou are an expert in performing set-operation in mathematics.",
            PROMPT_FINAL_ANSWER,
        ]
    ),
    # The system prompt instructs the LLM to recite its own
    # internal knowledge before answering the question
    "self_recitation": " ".join(
        [f"\n{PROMPT_THINKING}", PROMPT_FINAL_ANSWER]
    ),
    # The system prompt instructs the LLM to “think step-by-step”
    # to encourage it to reason through the problem
    "chain_of_thought": " ".join(
        ["\nThink step by step.", PROMPT_THINKING, PROMPT_FINAL_ANSWER]
    ),
    # The system prompt combines domain expertise, self-recitation,
    # chain-of-thought, and self-criticism.
    "composite": " ".join(
        [
            "\nYou are an expert in performing set-operation in mathematics.",
            "Think step by step.",
            PROMPT_THINKING,
            PROMPT_FINAL_ANSWER,
        ]
    ),
    # Composite - allow for empty set
    "composite_allow_empty": " ".join(
        [
            "\nThink step by step.",
            PROMPT_THINKING,
            PROMPT_FINAL_ANSWER,
            "The answer can be an empty set.",
        ]
    ),
}

# templates on guiding LLM to start with these values
PROMPT_TEMPLATES_ENDING = {
    "baseline": "",
    "baseline_allow_empty": "",
    "domain_expertise": "",
    "self_recitation": "<thinking>",
    "chain_of_thought": "<thinking>",
    "composite": "<thinking>",
    "composite_allow_empty": "<thinking>",
}

PROMPT_KSHOT_BEGIN = "\nThese are some examples:\n<examples>\n"
PROMPT_KSHOT_END = "</examples>\n"


def make_verb(members):
    """Make the prompt grammatically correct by defining the verb"""
    return "is" if len(members) == 1 else "are"


def make_set(members):
    """Convert set to a comma-delimited string, e.g. {1, 2, 3} -> "1, 2, 3" """
    if len(members) == 1:
        return str(list(members)[0])
    else:
        return ", ".join(str(m) for m in members)


def make_english_list(members):
    """Convert set to an English list, e.g. {1, 2, 3} -> "1, 2, and 3" """
    members = list(members)

    if len(members) == 0:
        raise ValueError("Empty set")
    elif len(members) == 1:
        return str(members[0])
    elif len(members) == 2:
        return " and ".join(str(m) for m in members)
    else:
        english_list = ", ".join(str(m) for m in members[:-1])
        english_list += ", and " + str(members[-1])
        return english_list


def make_rest_of_shot(operation, A, B, result):
    """Create the prompt examples based on the operation and inputs"""
    result_verb = make_verb(result)
    if operation == "union":
        return (
            f"print ({make_set(result)}), because {make_english_list(result)} "
            f"{result_verb} in either A or B."
        )
    elif operation == "intersection":
        if len(result):
            return (
                f"print ({make_set(result)}), because "
                f"{make_english_list(result)} {result_verb} "
                "in both A and B."
            )
        else:
            return (
                "print () (the empty set), because no members are in both A "
                "and B."
            )
    elif operation == "difference":
        if len(result):
            return (
                f"print ({make_set(result)}), because only "
                f"{make_english_list(result)} {result_verb} in A and not B."
            )
        else:
            return (
                "print () (the empty set), because all members of A are in B."
            )
    elif operation == "symmetric difference":
        only_in_A = A.intersection(result)
        only_in_B = B.intersection(result)
        only_in_B_verb = make_verb(only_in_B)
        rest_of_shot = f"print ({make_set(result)}), because "
        if len(only_in_A):
            only_in_A_verb = make_verb(only_in_A)
            rest_of_shot += (
                f"({make_set(only_in_A)}) {only_in_A_verb} in only A"
            )
            if len(only_in_B):
                rest_of_shot += " and "
        if len(only_in_B):
            only_in_B_verb = make_verb(only_in_B)
            rest_of_shot += (
                f"({make_set(only_in_B)}) {only_in_B_verb} in only B"
            )
        return rest_of_shot
    elif operation == "cartesian product":
        p = set(product(A, B))
        return (
            f"print ({make_set(p)}), because it is a set consisting of tuples "
            "formed from all pairs from A and B."
        )
    else:
        raise ValueError(f"Invalid operation {operation}")


def make_shot(operation, A, B):
    """Given the operation and two sets, creates a ground truth example"""
    result = get_ground_truth(operation, A, B)
    shot = f"If set A is ({make_set(A)}) and set B is ({make_set(B)}), "
    rest_of_shot = make_rest_of_shot(operation, A, B, result)
    return shot + rest_of_shot


def make_k_shot(prompt_config: PromptConfig):
    """Creates k-examples based on the operation and two sets
    (range of numbers are n, and the number of members are m)"""
    # return empty strings when k is 0
    if prompt_config.k_shot == 0:
        return ""

    # create the Sampler for k-shots
    k_shot_sampler = prompt_config.getKShot()

    shots = []
    for i in range(prompt_config.k_shot):
        A, B = k_shot_sampler[i]
        shots.append(make_shot(prompt_config.operation, A, B))
    k_shot = PROMPT_KSHOT_BEGIN
    if len(shots):
        for s in shots:
            k_shot += f"- {s}\n"
    k_shot += PROMPT_KSHOT_END
    return k_shot


def make_instruction_generator(experiment_type):
    """returns a function that generates the instruction based on
    the experiment type and prompt-language."""
    return {
        "formal_language": lambda op: OP_INSTRUCTION_FORMAL_LANGUAGE[op],
        "plain_language": lambda op: OP_INSTRUCTION_PLAIN_LANGUAGE[op],
        "functional_language": lambda op: OP_INSTRUCTION_FUNCTIONAL_STYLE[op],
        "pythonic_language": lambda op: OP_INSTRUCTION_PYTHONIC_CONCISENESS[
            op
        ],
        "iterative_accumulation": lambda op: OP_INSTRUCTION_ITERATIVE_ACCUMULATION[
            op
        ],
    }[experiment_type]


def get_ground_truth(operation, A, B):
    """returns the ground truth for the given operation and two sets"""
    func = OP_GROUND_TRUTH[operation]
    ground_truth = func(A, B)
    return ground_truth


def is_correct(ground_truth, result):
    """returns true if the result is the same as the ground truth"""
    return list(sorted(ground_truth)) == list(sorted(result))


def get_prompt(A, B, prompt_config, add_roles=False):
    """returns the prompt for the given instruction and two sets"""
    assert (
        prompt_config.approach in PROMPT_TEMPLATES.keys()
    ), f"the prompt approach of ({prompt_config.approach}) is not defined."
    A_str = ", ".join([str(a) for a in A])
    B_str = ", ".join([str(b) for b in B])

    # Add model-specific preamble
    if add_roles:
        prompt = "\n\nHuman: "
    else:
        prompt = ""

    # define the inputs and instruction
    prompt += (
        f"You are given two sets. Set A is ({A_str}). Set B is ({B_str})."
    )
    prompt += " You are given the following task:\n"
    prompt += f"<task> {prompt_config.get_instruction()} </task>"
    # add k-shot examples
    prompt += make_k_shot(prompt_config)
    # modify the prompt to test different capabilities (thinking, CoT, etc.)
    prompt += PROMPT_TEMPLATES[prompt_config.approach]
    # add model-specific ending
    if add_roles:
        prompt += f"\n\nAssistant: {PROMPT_TEMPLATES_ENDING[prompt_config.approach]}"
    else:
        prompt += f"\n\n{PROMPT_TEMPLATES_ENDING[prompt_config.approach]}"

    return prompt
