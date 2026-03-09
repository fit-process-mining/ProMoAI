from promoai.prompting.prompt_engineering import cut_conversation


def test_cut_history_1():
    conversation = [
        {
            "role": "user",
            "content": 'Your role: you are an expert in process modeling, familiar with common process constructs such as exclusive choice, do-redo loops, and partial orders. Your task is to analyze the textual description of a process and transform it into a process model in the POWL language. When generating a model, be as precise as possible and capture all details of the process in the model. Also act as the process owner and use your expertise and familiarity with the process context to fill in any missing knowledge. \n\nAdd…ed.Only two pools should be used: "Repairment Service" and "Customer". The "Repairment Service" pool should contain the lanes "Technicians", "Customer Service", and "Shipping Department", while the "Customer" pool should contain the lane "Customer". Splitting each lane into a different pool is a common mistake.\n\nAt the end of your response provide a single Python code snippet (i.e., staring with \'```python\') that contains the full final code. \n\nThis is the process description: A then B',
        },
        {
            "role": "assistant",
            "content": '```python\nfrom promoai.model_generation.generator import ModelGenerator\ngen = ModelGenerator()\na = gen.activity("A", pool = None, lane = None)\nb = gen.activity("B", pool = None, lane = None)\nfinal_model = gen.partial_order(dependencies=[(a, b)])\n```',
        },
        {
            "role": "user",
            "content": "Please update the model to fix it based on the provided feedback. Please make sure the returned model matches the initial process description, all previously provided feedback, and the newfeedback comment as well. Make sure to save the updated final model is the variable 'final_model'. This is the new feedback comment:add C",
        },
        {
            "role": "assistant",
            "content": '```python\nfrom promoai.model_generation.generator import ModelGenerator\ngen = ModelGenerator()\na = gen.activity("A", pool = None, lane = None)\nb = gen.activity("B", pool = None, lane = None)\nc = gen.activity("C", pool = None, lane = None)\nfinal_model = gen.partial_order(dependencies=[(a, b), (b, c)])\n```',
        },
        {
            "role": "user",
            "content": "Please update the model to fix it based on the provided feedback. Please make sure the returned model matches the initial process description, all previously provided feedback, and the newfeedback comment as well. Make sure to save the updated final model is the variable 'final_model'. This is the new feedback comment:add new pool and D in it",
        },
        {
            "role": "assistant",
            "content": '```python\nfrom promoai.model_generation.generator import ModelGenerator\ngen = ModelGenerator()\na = gen.activity("A", pool = None, lane = None)\nb = gen.activity("B", pool = None, lane = None)\nc = gen.activity("C", pool = None, lane = None)\nd = gen.activity("D", pool = "New Pool", lane = None)\nfinal_model = gen.partial_order(dependencies=[(a, b), (b, c), (c, d)])\n```',
        },
    ]
    index = 4
    cut_conv = cut_conversation(conversation, index)
    assert cut_conv[-1]["content"] == conversation[4]["content"]


def test_cut_history():
    conversation = [
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {
            "role": "user",
            "content": "Please update the model to fix it based on the provided feedback.",
            "type": "error",
        },
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "Can you tell me a joke?"},
        {
            "role": "assistant",
            "content": "Why did the scarecrow win an award? Because he was outstanding in his field!",
        },
        {"role": "user", "content": "That's funny! Tell me another one."},
        {
            "role": "user",
            "content": "Please update the model to fix it based on the provided feedback.",
        },
    ]
    index = 6
    cut_conv = cut_conversation(conversation, index)
    assert cut_conv[-1]["content"] == "That's funny! Tell me another one."
    index = -1
    # We expect an exception
    try:
        cut_conv = cut_conversation(conversation, index)
        assert False, "Expected an exception for negative index"
    except ValueError:
        pass
    index = 10
    # We expect an exception
    try:
        cut_conv = cut_conversation(conversation, index)
        assert False, "Expected an exception for out-of-bounds index"
    except ValueError:
        pass
    index = 4
    cut_conv = cut_conversation(conversation, index)
    assert cut_conv[-1]["content"] == "Can you tell me a joke?"
    index = 3
    cut_conv = cut_conversation(conversation, index)
    assert cut_conv[-1]["content"] == "Hello!"
