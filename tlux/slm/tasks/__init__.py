# mv logs/* sandbox_dir/* trash/ ; python3 __init__.py
# 
# ---------------------------------------------------------------------
#                        Handbook Follower Framework
# 
# The overarching objective of this framework is the provide the
# minimum possible scaffolding in order for locally operated language
# models to effectively follow a provided handbook and to autonomously
# consume work that it is provided. The handbook is expected to
# describe procedures, while a file (specific to the implementation)
# named 'tools.py' provides well-documented tools that are accessible
# to the model for completing assigned tasks. Once provided a source
# of work or objective, it continues to use the available tools to try
# to complete the goal or otherwise make progress.
# 
# Over time the handbook follower is expected to learn what
# personalizations and modifications to make to the handbook in order
# to prevent cyclic failures. In other words it will adapt to its
# working conditions.
# 
# ---------------------------------------------------------------------
# 
# Things to fix:
# 
#   The model has a tendency to get caught in loops. Needs to be
#   solved.  It also has a tendancy to claim failure when it *did not
#   fail*. Should instead prefer explicit failure signals from the
#   tools directly. (-1 fail, 0 neutral, 1 success)
# 
#   Reframe prompts about labeling steps take as "do they make
#   progress", because ultimate either learning how to fix something
#   or doing something new (not redundant with previous actions) is
#   useful.
# 
#   The sandbox should be versioned with actions. This will make it
#   easy to revert to a specific version with an injected "lesson"
#   context that informs the model how not to make the same mistakes
#   as before.
# 
#   For more complicated prompts, the model needs a way to "chunk"
#   problems and use `create_task` or similar to spawn a new worker
#   that sovles a specific subproblem. Paragraph level chunking seems
#   to be the most reliable. Hard to find better options.
# 
#   If the model writes tests that are bad, it tries to fix the
#   implementation instead of realizing it's the tests themselves
#   that are bad. The tests need to be written more carefully
#   maybe? How could this naturally be discovered?
# 
# Ideas to think about:
# 
#   A "convergent completion" is one that produces identical output
#   over many different rephrased versions of a prompt (that have
#   identical meaning).
# 
#   Paths to reliable responses:
#    + pick from options
#    + "explain thoughts" before picking
#    + "rephrase prompt", repeat above, look for consistency
#    ? request counter argument w/ binary approval
# 
#   Self-consistency under diverse input is an indicator of
#   correctness. Finding ways to identify, improve, or expose failures
#   in this is a very valuable signal.
# 
#   Exhaustive listing and trying all options is always a "brute
#   force" option.
# 
#   Finding ways to create, manage, and iterate over "mental lists" is
#   important for chunking.
# 
#   Reliability will increase with self evaluation beyond programming.
#   The model must be able to increase reliability where it observes
#   failures to get better.
# 
#   Introspective learning and self improvement happens by
#   occasionally reflecting on sequences of actions and deciding if
#   they are making progress.
# 
#   In order to improve performance when following the manual, there
#   needs to be a way to "learn from past mistakes". Which requires
#   recognizing if something was a mistake or not.
# 
#   How do we prevent repeating a mistake? How do we *identify* a
#   mistake? Is there a singular definition of mistake that we can use
#   here?
#     - lack of progress
#     - self repeating
# 
#   It's important to break a problem into parts, and then solve each
#   part independently while tracking some "shared lessons". Shared
#   lessons are things that will avoid mistakes or obtain solutions
#   that otherwise resulted in wasted effort.
# 


# Regular expressions for pulling language model outputs.
import os
import re
import time
from functools import partial

# Basic embedding model and language model for chat (step) completion.
from tlux.decorators import cache
# from tlux.embedding.client import get_embedding

# Import a function that lists all available tools.
from utilities import chat_complete, complete_from_options, rephrase
from tools import _get_tools, sandbox_dir
from versioning import snapshot_directory, diff_snapshots, undo_diff

# Handler for vector operations
import numpy as np

chat_complete = partial(chat_complete, temperature=0.5)

# Store snapshots of the sandbox directory (for undoing actions).
snapshots = [snapshot_directory(sandbox_dir)]

coder_manual = """
# Coder Manual

You will be given work in the form of an instruction or objective from a developer. You should author code to solve the problems provided by the developer. When writing code files, you should always also have tests for the code to verify (basic) correctness. Make sure to run all the tests and verify they pass before you claim completion.

## Best Practices

- When writing a python file, include a `if __name__ == "__main__":` block with a test call to the function and print statement to demonstrate correctness when run.
- Write extensive comments throughout code to describe what you are doing and make it readable and accessible.
- Consider performance and security when writing code; when there are known vulnerabilities, be sure to inform the developer.
- Always use unit tests (with mocks if appropriate) to verify the expected behavior of functions you author.
- When writing tests, focus on the most simple and clear cut cases to be sure the tests don't become too complicated.
- All unit tests need to be executed and unambiguously pass without errors.
- All tests should be contained in their own separate file that has the prefix "test_" before the existing file name.
- It is preferred for you to use only the standard library and not rely on external modules to solve problems.
"""


# Summarize the purpose of an individual that follows the given manual.
@cache()
def summarize_purpose(manual: str) -> str:
    purpose, _ = chat_complete(messages=[
        "In the next message you will receive an operating manual. Please acknowledge it only, further instruction will be provided.",
        "Okay.",
        manual,
        "Acknowledged.",
        "Produce a succinct description of the goals and purpose of an individual that follows the runbook provided above. Describe concisely what you think in a way that you would like it to be described if you were yourself instructed to follow that runbook."
    ])
    print("-"*70, flush=True)
    print("Purpose:")
    print()
    print(purpose)
    print()
    print("-"*70, flush=True)
    print()
    return purpose


@cache()
def extract_actions(manual: str) -> dict[str, tuple[str, list[tuple[str, str]]]]:
    # Split the manual into smaller sections (e.g., paragraphs)
    # sections = split_text_by_paragraph(manual)
    sections = _get_tools()
    actions = {}
    # Process each section individually
    for code_block in sections:
        # Now we are fairly certain this excerpt contains at least one action.
        # Extract all of the actions from the snippet.
        match = re.search(r"^def (\w+)\s*\(", code_block, re.MULTILINE)
        if match:
            function_name = match.group(1)  # Returns "run_file"
        else:
            continue

        # Get the function description.
        description, _ = chat_complete(
            messages=[
                f"```\n{code_block}\n```\n\nGiven the above code block as a reference, please provide a description for what the function `{function_name}` generally does. Respond with only the succinct description and nothing else.",
            ]
        )
        actions[function_name] = (description, [])
        # Extract all parameters from the current section.
        parameters, _ = chat_complete(
            messages=[
                f"```\n{code_block}\n```\n\nGiven the above code block, please extract the parameter names for the function `{function_name}`. Respond with each of the parameter names on a new line. Do not number, preface, nor explain the parameters. Only provide the parameters separated by new line characters. If `{function_name}` has no parameters, then simply respond with `None`.",
            ]
        )
        parameter_list = [
            p.strip() for p in parameters.strip().split("\n")
            if (len(p.strip()) > 0)
            and (not p.strip().lower() == "none")
            and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', p.strip())
        ]
        # Process each parameter to get its description
        for p in parameter_list:
            description, _ = chat_complete(
                messages=[
                    f"```\n{code_block}\n```\n\nGiven the above code block as a reference, please provide a description for the parameter `{p}` of the function `{function_name}`. Respond with only the succinct description of the parameter `{p}` and nothing else.",
                ]
            )
            actions[function_name][1].append((p, description))
    # Verification step to refine the list
    return actions


# Generate an action to take next.
def generate_action_for_state(
        manual: str,
        purpose: str,
        objective: str,
        history: list[str],
        actions: dict[str, tuple[str, list[tuple[str, str]]]],
        use_reasoning: bool = True,
) -> tuple[str, str]:
    # Compile the history into a more nicely formatted string.
    if len(history) > 0:
        history = "\n\n".join(history)
    else:
        history = "No actions have been taken yet."
    # Extrat plain text versions of all available actions.
    all_actions = ""
    for (action, (description, params)) in actions.items():
        all_actions += str(action) + "(" + (", ".join([p for (p,d) in params])) + f") - {description}\n"
    all_actions = all_actions.strip()
    # Strings for use in prompts.
    objective_block = f"\nProvided objective:\n```\n{objective}\n```\n"
    history_block = f"\nOperational history:\n```\n{history}\n```\n"
    action_block = f"\nAvailable actions:\n```\n{all_actions}\n```\n"
    # Generate a suggestion for what should be done next.
    manual_suggestion, _ = chat_complete(
        messages=[
            manual,
            "I see you have provided a manual. How can I help?",
            f"{objective_block}\n{history_block}\nFirst describe in your own words what has already been observed and accomplished, then describe what you think of as the best next step according to the manual."
        ],
    )
    suggested_addition = f"\n\nAccording to the operating manual, this is the suggestion for a next step:\n```\n{manual_suggestion.replace('```', '``')}\n```"
    # Generate thinking instructions.
    think_instruction = f"You have been given a high level objective and a history of your previous actions. At the end, you will be asked to pick one of the actions in order to advance towards the goal.\n{objective_block}\n{history_block}\n{action_block}\n{suggested_addition}\n\nGiven the objective and history of operations, describe what you think the general next step that should be taken and then which of the available actions are most likely to achieve that."
    messages = [think_instruction]
    if (use_reasoning):
        # Get the reasoning from the model for why it might pick any of the actions.
        reasoning, _ = chat_complete(prompt=think_instruction, system=purpose)
        messages += [reasoning, "Now please respond with the action you think is best to take first."]
    # Use chat_complete to generate candidate actions for a given step.
    chosen_action = complete_from_options(
        messages=messages,
        options=list(actions),
        system=purpose,
    )
    messages += [
        chosen_action,
        "Thank you. Now please succinctly state your intent in choosing this action so that future records may clearly show why it was chosen and what the purpose of choosing it was.",
    ]
    # Create an "intent" that can be recorded to the history.
    intent, _ = chat_complete(
        messages=messages,
        system=purpose,
    )
    # Store the chosen action as the "reasoning" if none was explicitly used.
    if (not use_reasoning):
        reasoning = intent
    # Return the chosen action and intent.
    return (chosen_action, intent, reasoning)


# Given the purpose, objective, history, documentation for actions, and chosen action,
#  construct arguments for the chosen action and then execute it. Summarize and return
#  the result of taking the action and return that summary as a string.
def execute_action(
        purpose: str,
        intent: str,
        history: list[str],
        actions: dict[str, tuple[str, list[tuple[str, str]]]],
        action: str
) -> tuple[dict[str, str], str, str]:
    import tools
    # If the action is invalid, 
    if action not in actions:
        return f"Failed to take action `{action}` because it is invalid. It is not in the available list of actions."
    # Construct contents for each argument of the action.
    history = "\n\n".join(history)
    parameter_values = {}
    action_description, parameters = actions[action]
    for parameter, description in parameters:
        value, _ = chat_complete(
            messages=[
                f"Operational history:\n```\n{history}\n```\n\nIntent:\n```\n{intent}\n```\n\nGiven our current operational history and intent, we are currently taking the action `{action}`. We need to provide contents for the parameter `{parameter}` with description `{description}`.\n\nPlease respond with what you believe the contents of the parameter `{parameter}` should be. Do not preface your response, do not include quotes or ticks, do not summarize your response. Only respond verbatim with what you intend to become the contents of the parameter '{parameter}'."
            ],
            system=purpose,
        )
        parameter_values[parameter] = value.strip()
    # Attempt to execute the action.
    try:
        action_function = getattr(tools, action)
        result = action_function(**parameter_values)
    except Exception as e:
        # Undo any changes in the sandbox directory if they were made.
        diff = diff_snapshots(snapshots[-1], snapshot_directory(sandbox_dir))
        if len(diff) > 0: undo_diff(sandbox_dir, diff)
        # Create an error traceback.
        import traceback
        # Get the full traceback and convert it to a string.
        traceback = traceback.format_exc()
        # Log the error message and the traceback.
        result = f"ERROR: {e}\n\nTraceback: {traceback}"
    parameter_values_string = ""
    for p in parameter_values:
        parameter_values_string += f"`{p}` was assigned the value:\n```\n{repr(parameter_values[p])}\n```\n\n"
    # Generate a summary of what occurred.
    summary, _ = chat_complete(
        messages=[
            f"Operational history:\n```\n{history}\n```\n\nIntent:\n```\n{intent}\n```\n\nGiven our current operational history and intent, we took the action `{action}`. The parameter values were as follows:\n\n{parameter_values_string}\n\nWe received the following result:\n```\n{result}\n```\n\nGiven that result we need to generate a one-line summary of what happened so that in the future we remember all necessary and important details learned from taking this action. Now, please respond with a succinct summary that can be inserted into the operational history for future reference. Do not provide a prefix or lead-in, simply respond with the single sentence that captures relevant information and succictly states what happened."
        ],
        system=purpose,
    )
    # Take a new snapshot of the directory.
    snapshots.append(snapshot_directory(sandbox_dir))
    # Return the parameter values used, the result, and the summary.
    return (parameter_values, result, summary)


# Return boolean indicating whether or not the current history appears
# to satisfy the requirements of the manual.
def satisfies_completion(manual: str, objective: str, history: list[str], checks: int = 3, reasoning_attempts: int = 1) -> bool:
    history = "\n\n".join(history)
    # Generate alternative phrasings of the question that marks conclusion.
    conclusion_questions = rephrase("Given the current history of operation, does it appear that the objective has been completely satisfied in accordance with all criteria listed in the manual?", new_phrasings=checks-1)
    # Each version of the question will produce a vote.
    votes = []
    for q in conclusion_questions:
        # Evaluate whether or not we are done.
        is_done = complete_from_options(
            reasoning_attempts=reasoning_attempts,
            messages=[
                manual,
                "I see you have provided a manual. How can I help?",
                f"Provided objective:\n```\n{objective}\n```\n\nOperational history:\n```\n{history}\n```\n\n" + q,
            ],
            options=["satisfied", "incomplete"],
        )
        votes.append(is_done == "satisfied")
    # Return "Yes" only if it is the most common answer.
    return (sum(votes) / len(votes) >= 0.5)


if __name__ == "__main__":
    # Define the specific objective
    objective = "Write a function to compute the Fibonacci sequence."
    # objective = "Read 'axy.f90' and generate a file with a list of all function names contained within 'axy.f90' and their purpose."
    reasoning_attempts = 0

    # Assign a local manual.
    manual = coder_manual
    # manual += f"\n## Tools\n\n```\n" + "\n\n".join(_get_tools()) + "\n```\n"

    # Get the purpose of the model (system prompt) for later "consistency".
    print("Consolidating purpose..", flush=True)
    print("", flush=True)
    purpose = summarize_purpose(manual)

    # Inspect the local 'tools.py' file tools that are available and generate descriptions.
    print("Extracting actions from manual..", flush=True)
    print("", flush=True)
    actions = extract_actions(manual)

    # Show all available actions.
    print("Actions: ")
    for a in actions:
        description, params = actions[a]
        print()
        print("", a + "(" + ", ".join([p for p,d in params]) + ")", "-", description)
        for p, d in params:
            print("  ", p, "-", d)
    print(flush=True)

    # Initalize a holder for history context.
    history: list[str] = []
    outcomes = {}
    lessons = {}
    while True:
        print("-"*50)

        # Generate an action for the current state using the SLM
        action, intent, reasoning = generate_action_for_state(manual, purpose, objective, history, actions)
        # Ensure that only our own custom instructions include code blocks.
        intent = intent.replace("```", "``")
        reasoning = reasoning.replace("```", "``")
        # Show the user the action, intent, and reasoning.
        print("Action:   ", repr(action), flush=True)
        print("Intent:   ", repr(intent), flush=True)
        print("Reasoning:", repr(reasoning), flush=True)
        history.append(intent.replace("```", "``"))

        # Execute the action and capture the outcome
        parameter_values, result, summary = execute_action(purpose, reasoning, history[:-1], actions, action)
        call_string = action + "("
        print()
        if len(parameter_values) > 0:
            print("Parameter_Values:")
            for (p,v) in parameter_values.items():
                if (len(v) > 70):
                    v = v[:70] + f"... remaining {len(v)-70} characters truncated ..."
                v = repr(v)
                print("", p, "=", v)
                call_string += "\n  " + f"{p} = {v},"
            call_string += "\n)"
        else:
            call_string += ")"
        print()
        print("Result: ", repr(result))
        print("Summary:", repr(summary))
        history.append(call_string.replace("```", "``"))
        history.append(summary.replace("```", "``"))

        # Label the "value" of the action, success, failure.
        label = complete_from_options(
            reasoning_attempts=reasoning_attempts,
            system=purpose,
            messages=[
                f"An action was taken with the following reason and intent.\n\nIntent:\n```\n{intent}\n```\n\nReasoning:\n```\n{reasoning}\n```\n\nThe action specifically that was chosen was '{action}'. After executing that action, the following outcome was observed:\n\nResult:\n```\n{result}\n```\n\nDo you think that outcome indicates success or failure against the original intent?",
            ],
            options=[
                "success",
                "failure"
            ]
        )
        outcomes[intent] = (label, result, summary)
        print("Label:  ", repr(label))
        # If the label is "failure", then create a "lesson" associated with this intent.
        if label == "failure":
            action_lessons = lessons.get(action, dict())
            lesson, _ = chat_complete(
                system=purpose,
                messages=[
                    f"An action was taken with the following reason and intent.\n\nIntent:\n```\n{intent}\n```\n\nReasoning:\n```\n{reasoning}\n```\n\nThe action specifically that was chosen was '{action}'. After executing that action, the following outcome was observed:\n\nResult:\n```\n{result}\n```\n\nDo you think that outcome indicates success or failure against the original intent?",
                    label,
                    "Please provide your best guess as to what would have prevented this from happening and resulted in a better outcome. Make your response as succinct as possible so it may be recorded and provided in the future when similar events occur.",
                ],
            )
            action_lessons[intent] = lesson
            print("Lesson: ", repr(lesson))              
        print()

        # If it appears the task has been completed according to the manual, break.
        if satisfies_completion(manual, objective, history, checks=1, reasoning_attempts=1):
            break

    # Once a terminal state is reached, conclude the process
    print("Task completed.")
