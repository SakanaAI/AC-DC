# Benchmark Dataset Structures

## Successfully Loaded

### 1. MMLU (`cais/mmlu`)
- **Subgroups**: 57 subjects (abstract_algebra, anatomy, astronomy, etc.)
- **Test samples**: 14,042
- **Columns**: `question`, `subject`, `choices` (list of 4 strings), `answer` (0-3)
- **Sampling**: 10 samples per subject = ~570 samples

**Example:**
```
question: "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
subject: "abstract_algebra"
choices: ['0', '4', '2', '6']
answer: 1 (index, meaning 'B' or '4')
```

### 2. MMLU Pro (`TIGER-Lab/MMLU-Pro`)
- **Subgroups**: 14 categories (biology, business, chemistry, etc.)
- **Test samples**: 12,032
- **Columns**: `question_id`, `question`, `options` (list of strings), `answer` (letter A-I), `answer_index`, `category`, `cot_content`, `src`
- **Sampling**: 10 samples per category = 140 samples

**Example:**
```
question: "Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence."
options: ['Safe practices, Fear, Jealousy, Trivial', 'Unsafe practices, Distress, Joy, Trivial', ...]
answer: "I"
answer_index: 8
category: "business"
```

### 3. GSM8K (`openai/gsm8k`)
- **Subgroups**: None
- **Test samples**: 1,319
- **Columns**: `question`, `answer`
- **Sampling**: 30 samples total

**Example:**
```
question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
answer: "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
```

### 4. HumanEval (`openai/openai_humaneval`)
- **Subgroups**: None
- **Test samples**: 164
- **Columns**: `task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`
- **Sampling**: 30 samples total (limited to 164 available)

**Example:**
```
task_id: "HumanEval/0"
prompt: "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
```

### 5. MBPP (`google-research-datasets/mbpp`)
- **Subgroups**: None
- **Test samples**: 500
- **Columns**: `task_id`, `text`, `code`, `test_list`, `test_setup_code`, `challenge_test_list`
- **Sampling**: 30 samples total

**Example:**
```
task_id: 11
text: "Write a python function to remove first and last occurrence of a given character from the string."
code: "def remove_Occ(s,ch): ..."
test_list: ['assert remove_Occ("hello","l") == "heo"', ...]
```

## Fixed with Alternatives

### 6. Big Bench Hard (`lukaemon/bbh`) ✓
- **Subgroups**: 27 tasks (boolean_expressions, causal_judgement, date_understanding, etc.)
- **Test samples per task**: ~250
- **Columns**: `input`, `target`
- **Sampling**: 10 samples per task = 270 samples

**Example:**
```
input: "not ( True ) and ( True ) is"
target: "False"
```

### 7. MATH Dataset (`EleutherAI/hendrycks_math`) ✓
- **Subgroups**: 7 categories (algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus)
- **Test samples**: Varies by category (~1000-1500 per category)
- **Columns**: `problem`, `level`, `type`, `solution`
- **Sampling**: 10 samples per category = 70 samples

**Example:**
```
problem: "How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?"
level: "Level 3"
type: "Algebra"
solution: "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$..."
```

## Skipped (Gated/Inaccessible)

### 8. GPQA Main
- **Status**: `Idavidrein/gpqa` is a **GATED DATASET** on HuggingFace
- **Action**: Skip for now unless user has access or provides alternative source
- **Note**: User should request access on HuggingFace if needed

## Question Formatting Strategy

### Multiple Choice (MMLU, MMLU Pro)
```
Question: [question text]
A) [choice 0]
B) [choice 1]
C) [choice 2]
D) [choice 3]
```

### Math Problems (GSM8K, MATH)
```
Problem: [question text]
```
(Don't include the solution in the embedding)

### Code Generation (HumanEval, MBPP)
```
[Problem description]

[Function signature/template if available]
```
(Don't include the solution in the embedding)

## Estimated Total Samples

- MMLU: 10 × 57 = 570
- MMLU Pro: 10 × 14 = 140
- Big Bench Hard: 10 × 27 = 270
- MATH: 10 × 7 = 70
- GSM8K: 30
- HumanEval: 30 (max available: 164)
- MBPP: 30
- GPQA: SKIPPED (gated dataset)

**Total**: ~1,110 samples across 7 benchmarks
