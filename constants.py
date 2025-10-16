
import random
from copy import deepcopy
from typing import List


ZS_PROMPT_TF = "SENTENCE\n\nAnswer with True or False:\nQUESTION\n\n"

OS_PROMPT_TF = """The doctor that the nurse called checked on the patient yesterday.

Answer with True or False:
The nurse called the doctor.
True


"""

TS_PROMPT_TF = """The doctor that the nurse called checked on the patient yesterday.

Answer with True or False:
The nurse called the doctor.
True


The teacher that helped the student graded the papers on the weekend.

Answer with True or False:
The student graded the papers.
False


"""

FS_PROMPT_TF = """The doctor that the nurse called checked on the patient yesterday.

Answer with True or False:
The nurse called the doctor.
True


The teacher that helped the student graded the papers on the weekend.

Answer with True or False:
The student graded the papers.
False


The sailor that the captain punished stayed in his room.

Answer with True or False:
The captain stayed in his room.
False


The driver that saved the cyclist went back home.

Answer with True or False:
The driver went back home.
True


"""

ES_PROMPT_TF = """The dentist that the children feared was, in reality, really gentle.

Answer with True or False:
The dentist was really gentle.
True


The singer that hired the guitarist arrived to the concert early.

Answer with True or False:
The singer hired the guitarist.
True


The professor that emailed the surgeon was stuck on a case.

Answer with True or False:
The surgeon email the professor.
False


The dog that the rescuers found stayed in the mountains for a week.

Answer with True or False:
The dog found the rescuers.
False


"""

PROMPTS_TF = {0: ZS_PROMPT_TF,
              1: OS_PROMPT_TF + ZS_PROMPT_TF,
              2: TS_PROMPT_TF + ZS_PROMPT_TF,
              4: FS_PROMPT_TF + ZS_PROMPT_TF,
              8: FS_PROMPT_TF + ES_PROMPT_TF + ZS_PROMPT_TF}


ZS_PROMPT_YN = "SENTENCE\n\nAnswer with Yes or No:\nQUESTION\n\n"

YN_1 = """The doctor that the nurse called checked on the patient yesterday.

Answer with Yes or No:
Did the nurse call the doctor?
My answer is: Yes"""


YN_2 = """The teacher that helped the student graded the papers on the weekend.

Answer with Yes or No:
Did the student grade the papers?
My answer is: No"""


YN_3 = """The sailor that the captain punished stayed in his room.

Answer with Yes or No:
Did the captain stay in his room?
My answer is: No"""


YN_4 = """The driver that saved the cyclist went back home.

Answer with Yes or No:
Did the driver go back home?
My answer is: Yes"""


YN_5 = """The dentist that the children feared was, in reality, really gentle.

Answer with Yes or No:
Was the dentist really gentle?
My answer is: Yes"""


YN_6 = """The singer that hired the guitarist arrived to the concert early.

Answer with Yes or No:
Did the singer hire the guitarist?
My answer is: Yes"""


YN_7 = """The professor that emailed the surgeon was stuck on a case.

Answer with Yes or No:
Did the surgeon email the professor?
My answer is: No"""


YN_8 = """The dog that the rescuers found stayed in the mountains for a week.

Answer with Yes or No:
Did the dog find the rescuers?
My answer is: No"""

EXAMPLES_YN = [YN_1, YN_2, YN_3, YN_4, YN_5, YN_6, YN_7, YN_8]


YN_1_REV = """Answer with Yes or No:
Did the nurse call the doctor?

Sentence: The doctor that the nurse called checked on the patient yesterday.

My answer is: Yes"""


YN_2_REV = """Answer with Yes or No:
Did the student grade the papers?

Sentence: The teacher that helped the student graded the papers on the weekend.

My answer is: No"""


YN_3_REV = """Answer with Yes or No:
Did the captain stay in his room?

Sentence: The sailor that the captain punished stayed in his room.

My answer is: No"""


YN_4_REV = """Answer with Yes or No:
Did the driver go back home?

Sentence: The driver that saved the cyclist went back home.

My answer is: Yes"""


YN_5_REV = """Answer with Yes or No:
Was the dentist really gentle?

Sentence: The dentist that the children feared was, in reality, really gentle.

My answer is: Yes"""


YN_6_REV = """Answer with Yes or No:
Did the singer hire the guitarist?

Sentence: The singer that hired the guitarist arrived to the concert early.

My answer is: Yes"""


YN_7_REV = """Answer with Yes or No:
Did the surgeon email the professor?

Sentence: The professor that emailed the surgeon was stuck on a case.

My answer is: No"""


YN_8_REV = """Answer with Yes or No:
Did the dog find the rescuers?

Sentence: The dog that the rescuers found stayed in the mountains for a week.

My answer is: No"""

EXAMPLES_YN_REV = [YN_1_REV, YN_2_REV, YN_3_REV, YN_4_REV, YN_5_REV, YN_6_REV, YN_7_REV, YN_8_REV]


ZS_PROMPT_YN_COT = "SENTENCE\n\nAnswer with Yes or No. Explain your answer first:\nQUESTION\n\n"

YN_1_COT = """The doctor that the nurse called checked on the patient yesterday.

Answer with Yes or No. Explain your answer first:
Did the nurse call the doctor?

Explanation: The relative clause “that the nurse called” assigns the nurse as agent and “doctor” as its object, so syntactic parsing confirms the nurse indeed called the doctor.

My answer is: Yes"""


YN_2_COT = """The teacher that helped the student graded the papers on the weekend.

Answer with Yes or No. Explain your answer first:
Did the student grade the papers?

Explanation: In the NP “the teacher that helped the student,” syntactic parsing assigns “teacher” as the agent of “graded,” and the relative clause only modifies the teacher’s identity, not the student’s.

My answer is: No"""


YN_3_COT = """The sailor that the captain punished stayed in his room.

Answer with Yes or No. Explain your answer first:
Did the captain stay in his room?

Explanation: In incremental parsing the main-clause subject (“the sailor”) carries the VP “stayed in his room,” and by pronoun‐antecedent bias “his” links to that subject, not to the embedded agent (“the captain”).

My answer is: No"""


YN_4_COT = """The driver that saved the cyclist went back home.

Answer with Yes or No. Explain your answer first:
Did the driver go back home?

Explanation: The main clause predicate “went back home” attaches to the subject NP “the driver,” so the driver is who went back home.

My answer is: Yes"""


YN_5_COT = """The dentist that the children feared was, in reality, really gentle.

Answer with Yes or No. Explain your answer first:
Was the dentist really gentle?

Explanation: The main clause “was … really gentle” directly attributes gentleness to the dentist despite the children’s fear.

My answer is: Yes"""


YN_6_COT = """The singer that hired the guitarist arrived to the concert early.

Answer with Yes or No. Explain your answer first:
Did the singer hire the guitarist?

Explanation: The restrictive relative clause “that hired the guitarist” syntactically assigns the agent role to “the singer,” so the singer is the one who hired the guitarist.

My answer is: Yes"""


YN_7_COT = """The professor that emailed the surgeon was stuck on a case.

Answer with Yes or No. Explain your answer first:
Did the surgeon email the professor?

Explanation: The relative clause “that emailed the surgeon” makes the professor the agent of emailing, so the surgeon was merely the recipient.

My answer is: No"""


YN_8_COT = """The dog that the rescuers found stayed in the mountains for a week.

Answer with Yes or No. Explain your answer first:
Did the dog find the rescuers?

Explanation: The relative clause “the rescuers found” assigns rescuers as the agents and the dog as the patient.

My answer is: No"""

EXAMPLES_YN_COT = [YN_1_COT, YN_2_COT, YN_3_COT, YN_4_COT, YN_5_COT, YN_6_COT, YN_7_COT, YN_8_COT]


YN_1_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the nurse call the doctor?

Sentence: The doctor that the nurse called checked on the patient yesterday.

Explanation: The relative clause “that the nurse called” assigns the nurse as agent and “doctor” as its object, so syntactic parsing confirms the nurse indeed called the doctor.

My answer is: Yes"""


YN_2_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the student grade the papers?

Sentence: The teacher that helped the student graded the papers on the weekend.

Explanation: In the NP “the teacher that helped the student,” syntactic parsing assigns “teacher” as the agent of “graded,” and the relative clause only modifies the teacher’s identity, not the student’s.

My answer is: No"""


YN_3_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the captain stay in his room?

Sentence: The sailor that the captain punished stayed in his room.

Explanation: In incremental parsing the main-clause subject (“the sailor”) carries the VP “stayed in his room,” and by pronoun‐antecedent bias “his” links to that subject, not to the embedded agent (“the captain”).

My answer is: No"""


YN_4_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the driver go back home?

Sentence: The driver that saved the cyclist went back home.

Explanation: The main clause predicate “went back home” attaches to the subject NP “the driver,” so the driver is who went back home.

My answer is: Yes"""


YN_5_COT_REV = """Answer with Yes or No. Explain your answer first:
Was the dentist really gentle?

Sentence: The dentist that the children feared was, in reality, really gentle.

Explanation: The main clause “was … really gentle” directly attributes gentleness to the dentist despite the children’s fear.

My answer is: Yes"""


YN_6_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the singer hire the guitarist?

Sentence: The singer that hired the guitarist arrived to the concert early.

Explanation: The restrictive relative clause “that hired the guitarist” syntactically assigns the agent role to “the singer,” so the singer is the one who hired the guitarist.

My answer is: Yes"""


YN_7_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the surgeon email the professor?

Sentence: The professor that emailed the surgeon was stuck on a case.

Explanation: The relative clause “that emailed the surgeon” makes the professor the agent of emailing, so the surgeon was merely the recipient.

My answer is: No"""


YN_8_COT_REV = """Answer with Yes or No. Explain your answer first:
Did the dog find the rescuers?

Sentence: The dog that the rescuers found stayed in the mountains for a week.

Explanation: The relative clause “the rescuers found” assigns rescuers as the agents and the dog as the patient.

My answer is: No"""

EXAMPLES_YN_COT_REV = [YN_1_COT_REV, YN_2_COT_REV, YN_3_COT_REV, YN_4_COT_REV, YN_5_COT_REV, YN_6_COT_REV, YN_7_COT_REV, YN_8_COT_REV]


ZS_PROMPT_Q = "SENTENCE\n\nAnswer this question in two words:\nQUESTION\n\n"

OS_PROMPT_Q = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
The doctor


"""

TS_PROMPT_Q = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
The doctor


The teacher that helped the student graded the papers on the weekend.

Answer this question:
Who helped the student?
The teacher


"""

QUESTION_1 = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
My answer is: The doctor
"""

QUESTION_2 = """The teacher went to the shop on Sunday.

Answer this question:
Who went to the shop?
My answer is: The teacher
"""

QUESTION_3 = """The sailor went to the captain's room.

Answer this question:
Who went to the captain's room?
My answer is: The sailor
"""

QUESTION_4 = """The driver that saved the cyclist went back home.

Answer this question:
Who saved the cyclist?
My answer is: The driver
"""

QUESTION_5 = """The dentist that the children feared was, in reality, really gentle.

Answer this question:
Who was really gentle?
My answer is: The dentist
"""

QUESTION_6 = """The singer that hired the guitarist arrived to the concert early.

Answer this question:
Who arrived at the concert early?
My answer is: The singer
"""

QUESTION_7 = """The professor emailed the surgeon on Monday.

Answer this question:
Who sent the email?
My answer is: The professor
"""

QUESTION_8 = """The dog that the rescuers found stayed in the mountains for a week.

Answer this question:
Who found the dog?
My answer is: The rescuers
"""

PROMPTS_Q = [QUESTION_1, QUESTION_2, QUESTION_3, QUESTION_4, QUESTION_5, QUESTION_6, QUESTION_7, QUESTION_8]


QUESTION_1_REV = """Answer this question:
Who did the nurse call?

Sentence: The doctor that the nurse called checked on the patient yesterday.

My answer is: The doctor
"""

QUESTION_2_REV = """Answer this question:
Who went to the shop?

Sentence: The teacher went to the shop on Sunday.

My answer is: The teacher
"""

QUESTION_3_REV = """Answer this question:
Who went to the captain's room?

Sentence: The sailor went to the captain's room.

My answer is: The sailor
"""

QUESTION_4_REV = """Answer this question:
Who saved the cyclist?

Sentence: The driver that saved the cyclist went back home.

My answer is: The driver
"""

QUESTION_5_REV = """Answer this question:
Who was really gentle?

Sentence: The dentist that the children feared was, in reality, really gentle.

My answer is: The dentist
"""

QUESTION_6_REV = """Answer this question:
Who arrived at the concert early?

Sentence: The singer that hired the guitarist arrived to the concert early.

My answer is: The singer
"""

QUESTION_7_REV = """Answer this question:
Who sent the email?

Sentence: The professor emailed the surgeon on Monday.

My answer is: The professor
"""

QUESTION_8_REV = """Answer this question:
Who found the dog?

Sentence: The dog that the rescuers found stayed in the mountains for a week.

My answer is: The rescuers
"""

PROMPTS_Q_REV = [QUESTION_1_REV, QUESTION_2_REV, QUESTION_3_REV, QUESTION_4_REV, QUESTION_5_REV, QUESTION_6_REV, QUESTION_7_REV, QUESTION_8_REV]

QUESTION_1_COT = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?

Explanation: Psycholinguistically, the relative clause “that the nurse called” makes “doctor” the object of “called,” showing the nurse called the doctor.

My answer is: The doctor
"""

QUESTION_2_COT = """The teacher went to the shop on Sunday.

Answer this question:
Who went to the shop?

Explanation: In English sentence processing, the initial noun phrase (“the teacher”) is parsed as the agent, so it’s interpreted as the one who went to the shop.

My answer is: The teacher
"""

QUESTION_3_COT = """The sailor went to the captain's room.

Answer this question:
Who went to the captain's room?

Explanation: The sailor. In the sentence 'The sailor went to the captain’s room,' 'the sailor' is the subject and thus the one who performed the action of going.

My answer is: The sailor
"""

QUESTION_4_COT = """The driver that saved the cyclist went back home.

Answer this question:
Who saved the cyclist?

Explanation: The driver saved the cyclist. The relative clause “that saved the cyclist” modifies “the driver,” identifying the driver as the rescuer.

My answer is: The driver
"""

QUESTION_5_COT = """The dentist that the children feared was, in reality, really gentle.

Answer this question:
Who was really gentle?

Explantion: The head noun 'the dentist' in the relative clause carries over as the subject of the main predicate, resolved by straightforward syntactic attachment rather than any intervening pronoun.

My answer is: The dentist
"""

QUESTION_6_COT = """The singer that hired the guitarist arrived to the concert early.

Answer this question:
Who arrived at the concert early?

Explanation: Psycholinguistic parsing favors attaching the main‐clause verb 'arrived' to the first noun phrase, so 'the singer' is interpreted as the subject of 'arrived.'

My answer is: The singer
"""

QUESTION_7_COT = """The professor emailed the surgeon on Monday.

Answer this question:
Who sent the email?

Explanation: In English’s SVO structure the subject noun phrase ('the professor') functions as the agent performing the action.

My answer is: The professor
"""

QUESTION_8_COT = """The dog that the rescuers found stayed in the mountains for a week.

Answer this question:
Who found the dog?

Explanation: Psycholinguistic parsing of the relative clause “that the rescuers found” assigns “the rescuers” as the agent who performed the action of finding.

My answer is: The rescuers
"""

PROMPTS_Q_COT = [QUESTION_1_COT, QUESTION_2_COT, QUESTION_3_COT, QUESTION_4_COT, QUESTION_5_COT, QUESTION_6_COT, QUESTION_7_COT, QUESTION_8_COT]


QUESTION_1_COT_REV = """Answer this question:
Who did the nurse call?

Sentence: The doctor that the nurse called checked on the patient yesterday.

Explanation: Psycholinguistically, the relative clause “that the nurse called” makes “doctor” the object of “called,” showing the nurse called the doctor.

My answer is: The doctor
"""

QUESTION_2_COT_REV = """Answer this question:
Who went to the shop?

Sentence: The teacher went to the shop on Sunday.

Explanation: In English sentence processing, the initial noun phrase (“the teacher”) is parsed as the agent, so it’s interpreted as the one who went to the shop.

My answer is: The teacher
"""

QUESTION_3_COT_REV = """Answer this question:
Who went to the captain's room?

Sentence: The sailor went to the captain's room.

Explanation: The sailor. In the sentence 'The sailor went to the captain’s room,' 'the sailor' is the subject and thus the one who performed the action of going.

My answer is: The sailor
"""

QUESTION_4_COT_REV = """Answer this question:
Who saved the cyclist?

Sentence: The driver that saved the cyclist went back home.

Explanation: Explanation: The driver saved the cyclist. The relative clause “that saved the cyclist” modifies “the driver,” identifying the driver as the rescuer.

My answer is: The driver
"""

QUESTION_5_COT_REV = """Answer this question:
Who was really gentle?

Sentence: The dentist that the children feared was, in reality, really gentle.

Explantion: The head noun 'the dentist' in the relative clause carries over as the subject of the main predicate, resolved by straightforward syntactic attachment rather than any intervening pronoun.

My answer is: The dentist
"""

QUESTION_6_COT_REV = """Answer this question:
Who arrived at the concert early?

Sentence: The singer that hired the guitarist arrived to the concert early.

Explanation: Psycholinguistic parsing favors attaching the main‐clause verb 'arrived' to the first noun phrase, so 'the singer' is interpreted as the subject of 'arrived.'

My answer is: The singer
"""

QUESTION_7_COT_REV = """Answer this question:
Who sent the email?

Sentence: The professor emailed the surgeon on Monday.

Explanation: In English’s SVO structure the subject noun phrase ('the professor') functions as the agent performing the action.

My answer is: The professor
"""

QUESTION_8_COT_REV = """Answer this question:
Who found the dog?

Sentence: The dog that the rescuers found stayed in the mountains for a week.

Explanation: Psycholinguistic parsing of the relative clause “that the rescuers found” assigns “the rescuers” as the agent who performed the action of finding.

My answer is: The rescuers
"""

PROMPTS_Q_COT_REV = [QUESTION_1_COT_REV, QUESTION_2_COT_REV, QUESTION_3_COT_REV, QUESTION_4_COT_REV, QUESTION_5_COT_REV, QUESTION_6_COT_REV, QUESTION_7_COT_REV, QUESTION_8_COT_REV]


FS_PROMPT_Q = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
The doctor


The teacher went to the shop on Sunday.

Answer this question:
Who went to the shop?
The teacher


The sailor went to the captain's room.

Answer this question:
Who did went to the captain's room?
The sailor


The driver that saved the cyclist went back home.

Answer this question:
Who saved the cyclist?
The driver


"""

ES_PROMPT_Q = """The dentist that the children feared was, in reality, really gentle.

Answer this question:
Who was really gentle?
The dentist


The singer that hired the guitarist arrived to the concert early.

Answer this question:
Who arrived at the concert early?
The singer


The professor emailed the surgeon on monday.

Answer this question:
Who sent the email?
The professor


The dog that the rescuers found stayed in the mountains for a week.

Answer this question:
Who found the dog?
The rescuers


"""


ZS_PROMPT_MC = "SENTENCE\n\nHere is the question:\nQUESTION\n\n"

OS_PROMPT_MC = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
A. The doctor
B. The nurse
I choose option A.


"""

TS_PROMPT_MC = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
A. The doctor
B. The nurse
I choose option A.


The teacher that helped the student graded the papers on the weekend.

Answer this question:
Who helped the student?
A. The paper
B. The teacher
I choose option B.


"""

FS_PROMPT_MC = sentence = """The doctor that the nurse called checked on the patient yesterday.

Answer this question:
Who did the nurse call?
A. The doctor
B. The nurse
I choose option A.


The teacher that helped the student graded the papers on the weekend.

Answer this question:
Who helped the student?
A. The paper
B. The teacher
I choose option B.


The sailor that the captain punished stayed in his room.

Answer this question:
Who did the captain punished?
A. The commander
B. The sailor
I choose option B.


The driver that saved the cyclist went back home.

Answer this question:
Who saved the cyclist?
A. The driver
B. The doctor
I choose option A.


"""

ES_PROMPT_MC = """The dentist that the children feared was, in reality, really gentle.

Answer this question:
Who was really gentle?
A. The dog
B. The dentist
I choose option B.


The singer that hired the guitarist arrived to the concert early contrary to the bass player.

Answer this question:
Who arrived at the concert early?
A. The singer
B. The bass player
I choose option A.


The professor that emailed the surgeon and the epidemiologist was stuck on a case.

Answer this question:
Who emailed the surgeon?
A. The professor
B. The epidemiologist
I choose option A.


The dog that the rescuers found stayed in the mountains for a week after fleeing from the hikers.

Answer this question:
Who found the dog?
A. The rescuers
B. The hikers
I choose option A.


"""

PROMPTS_MC = {0: "",
             1: OS_PROMPT_MC,
             2: TS_PROMPT_MC,
             4: FS_PROMPT_MC,
             8: FS_PROMPT_MC + ES_PROMPT_MC}

PROMPTS_MAPPING = {"yes_no": EXAMPLES_YN, "yes_no_cot": EXAMPLES_YN_COT, "true_false": PROMPTS_TF, "question": PROMPTS_Q, 
                    "yes_no_rev": EXAMPLES_YN_REV, "yes_no_cot_rev": EXAMPLES_YN_COT_REV, "question_rev": PROMPTS_Q_REV,
                    "question_cot": PROMPTS_Q_COT, "question_cot_rev": PROMPTS_Q_COT_REV}

ZS_MAPPING = {"yes_no": ZS_PROMPT_YN, "yes_no_cot": ZS_PROMPT_YN_COT, "true_false": ZS_PROMPT_TF, "question": ZS_PROMPT_Q}


DEFAULT_SYSTEM = """You are a linguistic experiment subject. You will be presented with a sentence, and will need to answer a \
reading comprehension question. You will need to select an option amongst the proposed answers.
Here are a few examples of questions and relevant answers:

EXAMPLES"""

DEFAULT_SYSTEM_2 = """You will answer a reading comprehension question about a sentence.
Here are a few examples of questions and correct answers:

EXAMPLES"""


DEFAULT_QUESTION = """Here is the sentence:
SENTENCE

Answer this question:
QUESTION"""


DEFAULT_QUESTION_REV = """Answer this question:
QUESTION

Here is the sentence:
SENTENCE"""


DEFAULT_QUESTION_COT = """Here is the sentence:
SENTENCE

Answer this question. Explain your answer first:
QUESTION"""

DEFAULT_QUESTION_COT_REV = """Answer this question. Explain your answer first:
QUESTION

Here is the sentence:
SENTENCE"""


PREFIX = """You are a linguistic experiment subject. You will be presented with a sentence, and will need to answer a \
reading comprehension question. You will need to select an option amongst the proposed answers.
Here are a few examples of questions and relevant answers:

EXAMPLES

Here is the sentence:
SENTENCE

Answer this question:
QUESTION
"""

OPTIONS = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M",
           14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S"}


def get_examples(question_type: str = "yes_no", num_examples: int = 4, num_samples: int = 8, suffix_type: str = "") -> List:

    """
    Returns shuffled lists of examples for a question type
    """

    suffix = "" if suffix_type == "" else f"_{suffix_type}"
    sampled_examples = list()
    for _ in range(num_samples):

        new_examples = deepcopy(PROMPTS_MAPPING[question_type + suffix])
        random.shuffle(new_examples)
        sampled_examples.append("\n\n\n".join(new_examples[:num_examples]))

    return sampled_examples