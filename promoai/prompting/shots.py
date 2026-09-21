"""
    This file includes example (description, model) pairs used for few-shots learning. Some process descriptions are
    from the PET data set "Patrizio Bellan, Han van der Aa, Mauro Dragoni, Chiara Ghidini, and Simone Paolo Ponzetto.
    PET: an annotated dataset for process extraction from natural language text tasks. In Business Process Management
    Workshops 2022, Revised Selected Papers, volume 460 of Lecture Notes in Business Information Processing,
    pages 315–321. Springer, 2022".
"""

from promoai.model_generation import ModelGenerator


# ---------------------------------------------------------------------------
# SHOT 1
# ---------------------------------------------------------------------------

d1 = (
    "in this process, you can either do 'a' or 'b'. If 'a' is selected,"
    " then it can be repeated. After completing 'a' or 'b', 'c' may be executed. c is always followed by 'd'."
    " Finally another"
    " execution of 'a' is performed. The whole process is optional and can be skipped."
)


def m1():
    gen = ModelGenerator()

    a = gen.activity("a")
    a_looped = gen.self_loop(a)
    a_copy = gen.copy(a)

    b = gen.activity("b")
    c = gen.activity("c")
    d = gen.activity("d")

    choice_a_b = gen.xor(a_looped, b)
    skippable_c_d = gen.skip(gen.sequence(c, d))

    final_model = gen.skip(
        gen.sequence(
            choice_a_b,
            skippable_c_d,
            a_copy,
        )
    )

    return final_model


def r_m1():
    gen = ModelGenerator()

    a = gen.activity("a", pool=None, lane=None)
    a_looped = gen.self_loop(a)
    a_copy = gen.copy(a)

    b = gen.activity("b", pool=None, lane=None)
    c = gen.activity("c", pool=None, lane=None)
    d = gen.activity("d", pool=None, lane=None)

    choice_a_b = gen.xor(a_looped, b)
    skippable_c_d = gen.skip(gen.sequence(c, d))

    final_model = gen.skip(
        gen.sequence(
            choice_a_b,
            skippable_c_d,
            a_copy,
        )
    )

    return final_model


e1 = (
    "A common error for this process is to add a dependency 'd -> a' without creating a "
    "copy of 'a'. This would model repetition instead of a distinct final execution of 'a'. "
    "Another common error is not to make the sequence c -> d skippable. "
    "The choice between a and b is a pure exclusive choice, so `xor` should be used, while the "
    "surrounding ordered behavior is naturally modeled with `sequence`."
)

r_e1 = (
    e1
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 1_2
# ---------------------------------------------------------------------------

d1_2 = (
    "in this process, you can either do 'a' or 'b'. If 'a' is selected,"
    " then it can be repeated. After completing 'a' or 'b', 'c' is executed, followed by 'd'. Finally, the process"
    " either ends or goes back to 'a'."
)


def m1_2():
    gen = ModelGenerator()

    a = gen.activity("a")
    a_looped = gen.self_loop(a)
    b = gen.activity("b")
    c = gen.activity("c")
    d = gen.activity("d")

    final_model = gen.decision_graph(
        dependencies=[
            (None, a_looped),
            (None, b),
            (a_looped, c),
            (b, c),
            (c, d),
            (d, a_looped),
            (d, None),
        ]
    )

    return final_model


def r_m1_2():
    gen = ModelGenerator()

    a = gen.activity("a", pool=None, lane=None)
    a_looped = gen.self_loop(a)
    b = gen.activity("b", pool=None, lane=None)
    c = gen.activity("c", pool=None, lane=None)
    d = gen.activity("d", pool=None, lane=None)

    final_model = gen.decision_graph(
        dependencies=[
            (None, a_looped),
            (None, b),
            (a_looped, c),
            (b, c),
            (c, d),
            (d, a_looped),
            (d, None),
        ]
    )

    return final_model


e1_2 = (
    "A common error for this process is to create a copy of 'a' instead of looping back to it. "
    "The dependency from d back to a makes the behavior cyclic, so the complete control flow should "
    "be modeled with a decision graph rather than decomposed into independent sequence or xor structures."
)

r_e1_2 = (
    e1_2
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 2
# ---------------------------------------------------------------------------

d2 = (
    "inventory management can proceed through restocking items or fulfilling orders. Restocking can be performed as "
    "often as necessary. Following either restocking or order fulfillment, an inventory audit is carried out. If"
    " unexpected behavior is detected in the inventory audit, then a data analysis is performed."
    " Additionally, urgent restocking needs can bypass regular restocking and "
    "order fulfillment processes directly leading to the inventory audit. This entire process is modular and can be "
    "repeated or skipped based on operational requirements."
)


def m2():
    gen = ModelGenerator()

    restock = gen.self_loop(gen.activity("restock items"))
    fulfil = gen.activity("fulfill orders")
    urgent_restock = gen.activity("urgent restock")

    inventory_audit = gen.activity("inventory audit")
    data_analysis = gen.activity("data analysis")

    process = gen.sequence(
        gen.xor(
            restock,
            fulfil,
            urgent_restock,
        ),
        inventory_audit,
        gen.skip(data_analysis),
    )

    final_model = gen.skip(gen.self_loop(process))

    return final_model


def r_m2():
    gen = ModelGenerator()

    restock = gen.self_loop(gen.activity("restock items", pool=None, lane=None))
    fulfil = gen.activity("fulfill orders", pool=None, lane=None)
    urgent_restock = gen.activity("urgent restock", pool=None, lane=None)

    inventory_audit = gen.activity("inventory audit", pool=None, lane=None)
    data_analysis = gen.activity("data analysis", pool=None, lane=None)

    process = gen.sequence(
        gen.xor(
            restock,
            fulfil,
            urgent_restock,
        ),
        inventory_audit,
        gen.skip(data_analysis),
    )

    final_model = gen.skip(gen.self_loop(process))

    return final_model


e2 = (
    "A common error is to copy 'inventory audit' or to construct an unnecessarily complex decision graph. "
    "The initial alternatives form a pure xor, followed by the inventory audit and an optional data analysis, "
    "so `xor`, `sequence`, and `skip` should be preferred. Another common mistake is to omit the self-loop "
    "on 'restock items' or the skippable self-loop on the whole process."
)

r_e2 = (
    e2
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 3
# ---------------------------------------------------------------------------

d3 = (
    "This enhanced payroll process allows for a high degree of customization and adaptation to specific "
    "requirements. Employees' time can be tracked with the option to repeat this step as needed. Pay calculations "
    "follows, incorporating diverse factors such as overtime, bonuses, and deductions. Subsequently, the process "
    "facilitates the issuance of payments and the generation of detailed reports. "
)


def m3():
    gen = ModelGenerator()

    track_time = gen.activity("track time")
    track_time_looped = gen.self_loop(track_time)

    calculate_pay = gen.activity("calculate pay")
    issue_payments = gen.activity("issue payments")
    generate_reports = gen.activity("generate reports")

    final_model = gen.sequence(
        track_time_looped,
        calculate_pay,
        gen.parallel(
            issue_payments,
            generate_reports,
        ),
    )

    return final_model


def r_m3():
    gen = ModelGenerator()

    track_time = gen.activity("track time", pool=None, lane=None)
    track_time_looped = gen.self_loop(track_time)

    calculate_pay = gen.activity("calculate pay", pool=None, lane=None)
    issue_payments = gen.activity("issue payments", pool=None, lane=None)
    generate_reports = gen.activity("generate reports", pool=None, lane=None)

    final_model = gen.sequence(
        track_time_looped,
        calculate_pay,
        gen.parallel(
            issue_payments,
            generate_reports,
        ),
    )

    return final_model


e3 = (
    "A common error for this process is to model a choice between issuing payments and generating reports. "
    "Both activities must happen and can proceed concurrently after pay has been calculated. Therefore the "
    "natural structure is a sequence ending in a parallel block, rather than a decision graph or xor."
)

r_e3 = (
    e3
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 4
# ---------------------------------------------------------------------------

d4 = (
    "This system combines 4 parallel subprocesses, i.e., that are executed independently/at the same time. The first "
    "process starts with A followed by B then a choice of C and D. The second process consists of a single activity "
    "E which can be repeated but must be executed at least once. The third process consists of the activity F, "
    "which can be repeated or skipped. The last process contains the parallel activities G, H, I, J with the "
    "constrains that I must precede J and H must precede I "
)


def m4():
    gen = ModelGenerator()

    # subprocess 1
    a = gen.activity("a")
    b = gen.activity("b")
    c = gen.activity("c")
    d = gen.activity("d")

    subprocess_1 = gen.sequence(
        a,
        b,
        gen.xor(c, d),
    )

    # subprocess 2
    e = gen.activity("e")
    subprocess_2 = gen.self_loop(e)

    # subprocess 3
    f = gen.activity("f")
    subprocess_3 = gen.skip(gen.self_loop(f))

    # subprocess 4
    g = gen.activity("g")
    h = gen.activity("h")
    i = gen.activity("i")
    j = gen.activity("j")

    subprocess_4 = gen.parallel(
        g,
        gen.sequence(h, i, j),
    )

    # all four subprocesses execute concurrently
    final_model = gen.parallel(
        subprocess_1,
        subprocess_2,
        subprocess_3,
        subprocess_4,
    )

    return final_model


def r_m4():
    gen = ModelGenerator()

    # subprocess 1
    a = gen.activity("a", pool=None, lane=None)
    b = gen.activity("b", pool=None, lane=None)
    c = gen.activity("c", pool=None, lane=None)
    d = gen.activity("d", pool=None, lane=None)

    subprocess_1 = gen.sequence(
        a,
        b,
        gen.xor(c, d),
    )

    # subprocess 2
    e = gen.activity("e", pool=None, lane=None)
    subprocess_2 = gen.self_loop(e)

    # subprocess 3
    f = gen.activity("f", pool=None, lane=None)
    subprocess_3 = gen.skip(gen.self_loop(f))

    # subprocess 4
    g = gen.activity("g", pool=None, lane=None)
    h = gen.activity("h", pool=None, lane=None)
    i = gen.activity("i", pool=None, lane=None)
    j = gen.activity("j", pool=None, lane=None)

    subprocess_4 = gen.parallel(
        g,
        gen.sequence(h, i, j),
    )

    final_model = gen.parallel(
        subprocess_1,
        subprocess_2,
        subprocess_3,
        subprocess_4,
    )

    return final_model


e4 = (
    "A common error is to model the four subprocesses as alternatives instead of concurrent behavior. "
    "All four subprocesses must be executed, so the top-level structure is `parallel`. "
    "Within the first subprocess, C and D form a pure exclusive choice and should use `xor`. "
    "Within the fourth subprocess, G is independent of the ordered sequence H -> I -> J."
)

r_e4 = (
    e4
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 5
# ---------------------------------------------------------------------------

d5 = (
    "A customer brings in a defective computer and the CRS checks the defect and hands out a repair cost calculation "
    "back. If the customer decides that the costs are acceptable , the process continues , otherwise she takes her "
    "computer home unrepaired. The ongoing repair consists of two activities , which are executed , in an arbitrary "
    "order. The first activity is to check and repair the hardware , whereas the second activity checks and "
    "configures the software. After each of these activities , the proper system functionality is tested. If an "
    "error is detected another arbitrary repair activity is executed , otherwise the repair is finished. "
)


def m5():
    gen = ModelGenerator()

    defect_check = gen.activity("Check defect")
    cost_calculation = gen.activity("Calculate repair costs")

    cancel = gen.activity("Cancel and give computer unrepaired")

    repair_hardware = gen.activity("Check and repair the hardware")
    repair_software = gen.activity("Check and configure the software")

    test_after_hardware = gen.activity("Test system functionality")
    test_after_software = gen.activity("Test system functionality")

    additional_hardware_repair = gen.activity("Perform additional hardware repairs")
    additional_software_repair = gen.activity("Perform additional software repairs")

    finish_repair = gen.activity("Finish repair")

    hardware_branch = gen.sequence(
        repair_hardware,
        test_after_hardware,
        gen.skip(additional_hardware_repair),
    )

    software_branch = gen.sequence(
        repair_software,
        test_after_software,
        gen.skip(additional_software_repair),
    )

    repair_path = gen.sequence(
        gen.parallel(
            hardware_branch,
            software_branch,
        ),
        finish_repair,
    )

    final_model = gen.sequence(
        defect_check,
        cost_calculation,
        gen.xor(
            cancel,
            repair_path,
        ),
    )

    return final_model


def r_m5():
    gen = ModelGenerator()

    defect_check = gen.activity(
        "Check defect",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    cost_calculation = gen.activity(
        "Calculate repair costs",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    cancel = gen.activity(
        "Cancel",
        pool="Customer",
        lane="Customer",
    )

    return_computer = gen.activity(
        "Give computer unrepaired",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    repair_hardware = gen.activity(
        "Check and repair the hardware",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    repair_software = gen.activity(
        "Check and configure the software",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    test_after_hardware = gen.activity(
        "Test system functionality",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    test_after_software = gen.activity(
        "Test system functionality",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    additional_hardware_repair = gen.activity(
        "Perform additional hardware repairs",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    additional_software_repair = gen.activity(
        "Perform additional software repairs",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    finish_repair = gen.activity(
        "Finish repair",
        pool="Repairment Service",
        lane="Customer Repair Specialist",
    )

    cancel_path = gen.sequence(
        cancel,
        return_computer,
    )

    hardware_branch = gen.sequence(
        repair_hardware,
        test_after_hardware,
        gen.skip(additional_hardware_repair),
    )

    software_branch = gen.sequence(
        repair_software,
        test_after_software,
        gen.skip(additional_software_repair),
    )

    repair_path = gen.sequence(
        gen.parallel(
            hardware_branch,
            software_branch,
        ),
        finish_repair,
    )

    final_model = gen.sequence(
        defect_check,
        cost_calculation,
        gen.xor(
            cancel_path,
            repair_path,
        ),
    )

    return final_model


e5 = (
    "A common error is to model the hardware and software branches as an exclusive choice. "
    "Both branches must happen, although they may proceed independently, so they should be combined with `parallel`. "
    "Another important error is to model a local choice between 'cancel' and an artificial 'continue' activity. "
    "The actual exclusive choice is between the cancellation path and the complete repair path."
)

r_e5 = (
    e5
    + " Another common error is to create more pools and lanes. Only two pools should be used: "
    '"Repairment Service" and "Customer". The "Repairment Service" pool should contain the lane '
    '"Customer Repair Specialist", while the "Customer" pool should contain the lane "Customer".'
)


# ---------------------------------------------------------------------------
# SHOT 6
# ---------------------------------------------------------------------------

d6 = (
    "A small company manufactures customized bicycles. Whenever the sales department receives an order , "
    "a new process instance is created. A member of the sales department can then reject or accept the order "
    "for a customized bike. In the former case , the process instance is finished. In the latter case , "
    "the storehouse and the engineering department are informed. The storehouse immediately processes the part "
    "list of the order and checks the required quantity of each part. If the part is available in-house , "
    "it is reserved. If it is not available , it is back-ordered. This procedure is repeated for each item on "
    "the part list. In the meantime, the engineering department prepares everything for the assembling of the "
    "ordered bicycle. If the storehouse has successfully reserved or back-ordered every item of the part list "
    "and the preparation activity has finished, the engineering department assembles the bicycle. Afterwards "
    ", the sales department ships the bicycle to the customer and finishes the process instance . "
)


def m6():
    gen = ModelGenerator()

    create_process = gen.activity("Create process instance")
    reject_order = gen.activity("Reject order")
    accept_order = gen.activity("Accept order")

    inform = gen.activity("Inform storehouse and engineering department")
    process_part_list = gen.activity("Process part list")

    check_part = gen.activity("Check required quantity of the part")
    reserve = gen.activity("Reserve part")
    back_order = gen.activity("Back-order part")

    prepare_assembly = gen.activity("Prepare bicycle assembly")
    assemble_bicycle = gen.activity("Assemble bicycle")
    ship_bicycle = gen.activity("Ship bicycle")
    finish_process = gen.activity("Finish process instance")

    part_subprocess = gen.sequence(
        check_part,
        gen.xor(
            reserve,
            back_order,
        ),
    )

    part_subprocess = gen.self_loop(part_subprocess)

    concurrency = gen.parallel(
        part_subprocess,
        prepare_assembly,
    )

    final_model = gen.decision_graph(
        dependencies=[
            (None, create_process),
            (create_process, reject_order),
            (create_process, accept_order),
            (reject_order, finish_process),
            (accept_order, inform),
            (inform, process_part_list),
            (process_part_list, concurrency),
            (concurrency, assemble_bicycle),
            (assemble_bicycle, ship_bicycle),
            (ship_bicycle, finish_process),
            (finish_process, None),
        ]
    )

    return final_model


def r_m6():
    gen = ModelGenerator()

    create_process = gen.activity(
        "Create process instance",
        pool="Bike manufacturing Company",
        lane="Sales Department",
    )

    reject_order = gen.activity(
        "Reject order",
        pool="Bike manufacturing Company",
        lane="Sales Department",
    )

    accept_order = gen.activity(
        "Accept order",
        pool="Bike manufacturing Company",
        lane="Sales Department",
    )

    inform = gen.activity(
        "Inform storehouse and engineering department",
        pool="Bike manufacturing Company",
        lane="Sales Department",
    )

    process_part_list = gen.activity(
        "Process part list",
        pool="Bike manufacturing Company",
        lane="Storehouse",
    )

    check_part = gen.activity(
        "Check required quantity of the part",
        pool="Bike manufacturing Company",
        lane="Storehouse",
    )

    reserve = gen.activity(
        "Reserve part",
        pool="Bike manufacturing Company",
        lane="Storehouse",
    )

    back_order = gen.activity(
        "Back-order part",
        pool="Bike manufacturing Company",
        lane="Storehouse",
    )

    prepare_assembly = gen.activity(
        "Prepare bicycle assembly",
        pool="Bike manufacturing Company",
        lane="Engineering Department",
    )

    assemble_bicycle = gen.activity(
        "Assemble bicycle",
        pool="Bike manufacturing Company",
        lane="Engineering Department",
    )

    ship_bicycle = gen.activity(
        "Ship bicycle",
        pool="Bike manufacturing Company",
        lane="Sales Department",
    )

    finish_process = gen.activity(
        "Finish process instance",
        pool="Bike manufacturing Company",
        lane="Sales Department",
    )

    part_subprocess = gen.sequence(
        check_part,
        gen.xor(
            reserve,
            back_order,
        ),
    )

    part_subprocess = gen.self_loop(part_subprocess)

    concurrency = gen.parallel(
        part_subprocess,
        prepare_assembly,
    )

    final_model = gen.decision_graph(
        dependencies=[
            (None, create_process),
            (create_process, reject_order),
            (create_process, accept_order),
            (reject_order, finish_process),
            (accept_order, inform),
            (inform, process_part_list),
            (process_part_list, concurrency),
            (concurrency, assemble_bicycle),
            (assemble_bicycle, ship_bicycle),
            (ship_bicycle, finish_process),
            (finish_process, None),
        ]
    )

    return final_model


e6 = (
    "A common error is not to model the concurrency between the repeated part-handling subprocess and "
    "the preparation of the assembly. These two branches must both happen and are independent, so `parallel` "
    "should be used. Within the part-handling subprocess, reserving and back-ordering are alternatives and "
    "should therefore be modeled with `xor`."
)

r_e6 = (
    e6
    + ' There is exactly one organization here: "Bike manufacturing Company". There are three lanes: '
    '"Sales Department", "Storehouse", and "Engineering Department". Splitting each lane into a different '
    "pool is a common mistake."
)


# ---------------------------------------------------------------------------
# SHOT 7
# ---------------------------------------------------------------------------

d7 = (
    "A and B can happen in any order (concurrent). C and D can happen in any order. A precedes both C and D. B "
    "precedes D"
)


def m7():
    gen = ModelGenerator()

    a = gen.activity("A")
    b = gen.activity("B")
    c = gen.activity("C")
    d = gen.activity("D")

    final_model = gen.partial_order(
        dependencies=[
            (a, c),
            (a, d),
            (b, d),
        ]
    )

    return final_model


def r_m7():
    gen = ModelGenerator()

    a = gen.activity("A", pool=None, lane=None)
    b = gen.activity("B", pool=None, lane=None)
    c = gen.activity("C", pool=None, lane=None)
    d = gen.activity("D", pool=None, lane=None)

    final_model = gen.partial_order(
        dependencies=[
            (a, c),
            (a, d),
            (b, d),
        ]
    )

    return final_model


e7 = (
    "A common error is to model A and B as one parallel block and C and D as another parallel block, then "
    "sequence these two blocks. That would incorrectly introduce the dependency B -> C. "
    "The actual behavior contains irregular precedence constraints: A -> C, A -> D, and B -> D, while B and C "
    "remain independent. A direct partial order is therefore the appropriate representation."
)

r_e7 = (
    e7
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 8
# ---------------------------------------------------------------------------

d8 = (
    "A followed by B or C. Then D or G. A can be followed by G. "
    "Optionally, after A, we can skip all other activities."
)


def m8():
    gen = ModelGenerator()

    a = gen.activity("A")
    b = gen.activity("B")
    c = gen.activity("C")
    d = gen.activity("D")
    g = gen.activity("G")

    final_model = gen.decision_graph(
        dependencies=[
            (None, a),
            (a, b),
            (a, g),
            (a, c),
            (b, d),
            (b, g),
            (c, g),
            (c, d),
            (d, None),
            (g, None),
            (a, None),
        ]
    )

    return final_model


def r_m8():
    gen = ModelGenerator()

    a = gen.activity("A", pool=None, lane=None)
    b = gen.activity("B", pool=None, lane=None)
    c = gen.activity("C", pool=None, lane=None)
    d = gen.activity("D", pool=None, lane=None)
    g = gen.activity("G", pool=None, lane=None)

    final_model = gen.decision_graph(
        dependencies=[
            (None, a),
            (a, b),
            (a, g),
            (a, c),
            (b, d),
            (b, g),
            (c, g),
            (c, d),
            (d, None),
            (g, None),
            (a, None),
        ]
    )

    return final_model


e8 = (
    "A common error is to decompose this behavior into unnecessary xor, sequence, or partial-order structures. "
    "The available alternatives depend on the path already taken, and A may also terminate the process directly. "
    "One decision graph should therefore model the complete path-dependent control flow."
)

r_e8 = (
    e8
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 9
# ---------------------------------------------------------------------------

d9 = (
    "The process starts with checking part stock availability. "
    "After that, we either cancel the order because we don't have the required parts to produce it, "
    "or we continue with the production. When we cancel the order, we notify the customer via e-mail. "
    "If we can produce the machines, we check the production schedule and "
    "we schedule the production of each machine. In the end, we produce the machines. "
    "As soon as the machines are produced, we ship the machines. In the end, we notify the customer via e-mail or via the system."
)


def m9():
    gen = ModelGenerator()

    check_schedule = gen.activity("Check production schedule")
    schedule_production = gen.activity("Schedule production")
    produce_machines = gen.activity("Produce machines")
    ship_machines = gen.activity("Ship machines")

    production_block = gen.sequence(
        check_schedule,
        schedule_production,
        produce_machines,
        ship_machines,
    )

    check_stock = gen.activity("Check part stock availability")
    cancel = gen.activity("Cancel order")
    notify_email = gen.activity("Notify via email")
    notify_system = gen.activity("Notify via system")

    final_model = gen.decision_graph(
        dependencies=[
            (None, check_stock),
            (check_stock, cancel),
            (check_stock, production_block),
            (cancel, notify_email),
            (production_block, notify_email),
            (production_block, notify_system),
            (notify_email, None),
            (notify_system, None),
        ]
    )

    return final_model


def r_m9():
    gen = ModelGenerator()

    check_schedule = gen.activity(
        "Check production schedule",
        pool=None,
        lane=None,
    )
    schedule_production = gen.activity(
        "Schedule production",
        pool=None,
        lane=None,
    )
    produce_machines = gen.activity(
        "Produce machines",
        pool=None,
        lane=None,
    )
    ship_machines = gen.activity(
        "Ship machines",
        pool=None,
        lane=None,
    )

    production_block = gen.sequence(
        check_schedule,
        schedule_production,
        produce_machines,
        ship_machines,
    )

    check_stock = gen.activity(
        "Check part stock availability",
        pool=None,
        lane=None,
    )
    cancel = gen.activity(
        "Cancel order",
        pool=None,
        lane=None,
    )
    notify_email = gen.activity(
        "Notify via email",
        pool=None,
        lane=None,
    )
    notify_system = gen.activity(
        "Notify via system",
        pool=None,
        lane=None,
    )

    final_model = gen.decision_graph(
        dependencies=[
            (None, check_stock),
            (check_stock, cancel),
            (check_stock, production_block),
            (cancel, notify_email),
            (production_block, notify_email),
            (production_block, notify_system),
            (notify_email, None),
            (notify_system, None),
        ]
    )

    return final_model


e9 = (
    "A common mistake is to model the exclusive paths as partial orders of independent xor structures. "
    "The production activities themselves form a pure sequence and should use `sequence`. "
    "The surrounding behavior is path-dependent: cancellation can only lead to email notification, while "
    "successful production can lead to either email or system notification. Therefore the surrounding behavior "
    "should remain one decision graph."
)

r_e9 = (
    e9
    + " Another common error is to assign lanes and pools as there are no roles or organizations mentioned in the description."
)


# ---------------------------------------------------------------------------
# SHOT 10
# ---------------------------------------------------------------------------

"""
Machine repairment service, the example is taken from "Activity Instance Identification using Bipartite
Graph Matching", C-Y Li et al.
"""

d10 = (
    "A process for equipment repair and maintenance service is observed."
    "First, a customer sends a machine for repairment (SendMachine)."
    "As soon as the machine is received by the service, the repair is registered (RegisterRepair)."
    "Then, the technicians analyze the defects (AnalyzeDefects), while the customer service department checks the warranty"
    "(CheckWarranty)."
    "Afterwards, if a repair is possible, first, the technicians dismantle the machine (DismantleMachine)"
    "and then repair the faulty parts (RepairPart). If a repair is infeasible,"
    "then the customer service orders a new machine (OrderMachine)."
    "In the end, the repaired or the new machine is shipped back"
    "to the customer by the shipping department (ShipBack)."
)


def r_m10():
    gen = ModelGenerator()

    send_machine = gen.activity(
        "SendMachine",
        pool="Customer",
        lane="Customer",
    )

    register_repair = gen.activity(
        "RegisterRepair",
        pool="RepairService",
        lane="Customer Service",
    )

    analyze_defects = gen.activity(
        "AnalyzeDefects",
        pool="RepairService",
        lane="Technicians",
    )

    check_warranty = gen.activity(
        "CheckWarranty",
        pool="RepairService",
        lane="Customer Service",
    )

    dismantle_machine = gen.activity(
        "DismantleMachine",
        pool="RepairService",
        lane="Technicians",
    )

    repair_part = gen.activity(
        "RepairPart",
        pool="RepairService",
        lane="Technicians",
    )

    order_machine = gen.activity(
        "OrderMachine",
        pool="RepairService",
        lane="Customer Service",
    )

    ship_back = gen.activity(
        "ShipBack",
        pool="RepairService",
        lane="Shipping Department",
    )

    repair_machine = gen.sequence(
        dismantle_machine,
        gen.self_loop(repair_part),
    )

    repair_or_order = gen.xor(
        repair_machine,
        order_machine,
    )

    final_model = gen.sequence(
        send_machine,
        register_repair,
        gen.parallel(
            analyze_defects,
            check_warranty,
        ),
        repair_or_order,
        ship_back,
    )

    return final_model


r_e10 = (
    "AnalyzeDefects and CheckWarranty can be executed concurrently, but both must complete before the "
    "repair-or-replace choice. Therefore `parallel` should be used for these two activities. "
    "Repairing the machine consists of the sequence DismantleMachine followed by one or more executions "
    "of RepairPart, while repairing and ordering a new machine are mutually exclusive alternatives and "
    "should be modeled with `xor`. "
    'Only two pools should be used: "RepairService" and "Customer". The "RepairService" pool should contain '
    'the lanes "Technicians", "Customer Service", and "Shipping Department", while the "Customer" pool should '
    'contain the lane "Customer". Splitting each lane into a different pool is a common mistake.'
)


# d11 = (
#     "This process begins when a customer signs up for a subscription service,"
#     "entering personal and payment information. The system generates an account"
#     ", assigns access, and triggers automated billing cycles."
#     "Throughout the subscription, the customer receives regular updates,"
#     "product enhancements, or renewal notifications. If the customer decides to cancel,"
#     "they submit a cancellation request, which the support team processes. "
#     "Depending on the terms, any refunds or charges are applied."
#     "The process concludes when the subscription is deactivated by the support team"
#     " and the final account balance is settled."
# )
#
#
# def r_m11():
#     gen = ModelGenerator()
#
#     sign_up = gen.activity(
#         "Sign up",
#         pool="Customer",
#         lane="Customer",
#     )
#
#     generate_account = gen.activity(
#         "Generate account",
#         pool="Subscription Service",
#         lane="System",
#     )
#
#     assign_access = gen.activity(
#         "Assign access",
#         pool="Subscription Service",
#         lane="System",
#     )
#
#     trigger_billing = gen.activity(
#         "Trigger billing",
#         pool="Subscription Service",
#         lane="System",
#     )
#
#     send_updates = gen.activity(
#         "Send updates and notifications",
#         pool="Subscription Service",
#         lane="System",
#     )
#
#     submit_cancellation = gen.activity(
#         "Submit cancellation request",
#         pool="Customer",
#         lane="Customer",
#     )
#
#     process_cancellation = gen.activity(
#         "Process cancellation",
#         pool="Subscription Service",
#         lane="Support Team",
#     )
#
#     apply_refunds_charges = gen.activity(
#         "Apply refunds or charges",
#         pool="Subscription Service",
#         lane="Support Team",
#     )
#
#     deactivate_subscription = gen.activity(
#         "Deactivate subscription",
#         pool="Subscription Service",
#         lane="Support Team",
#     )
#
#     settle_final_balance = gen.activity(
#         "Settle final account balance",
#         pool="Subscription Service",
#         lane="Support Team",
#     )
#
#     billing_loop = gen.self_loop(trigger_billing)
#     update_loop = gen.self_loop(send_updates)
#
#     cancellation_path = gen.skip(
#         gen.sequence(
#             submit_cancellation,
#             process_cancellation,
#             gen.parallel(
#                 gen.skip(apply_refunds_charges),
#                 deactivate_subscription,
#             ),
#             settle_final_balance,
#         )
#     )
#
#     final_model = gen.sequence(
#         sign_up,
#         generate_account,
#         assign_access,
#         gen.parallel(
#             billing_loop,
#             update_loop,
#         ),
#         cancellation_path,
#     )
#
#     return final_model
#
#
# r_e11 = (
#     "A common mistake is to omit the loops for the billing cycle and updates/notifications, "
#     "as these are recurring activities throughout the subscription period. "
#     'Only two pools should be used: "Subscription Service" and "Customer". '
#     'The "Subscription Service" pool should contain the lanes "System" and "Support Team", '
#     'while the "Customer" pool should contain the lane "Customer". '
#     "Putting each lane into a different pool is a common mistake."
# )


SHOTS = [
    (d1, m1, e1),
    (d1_2, m1_2, e1_2),
    (d2, m2, e2),
    (d3, m3, e3),
    (d4, m4, e4),
    (d5, m5, e5),
    (d6, m6, e6),
    (d7, m7, e7),
    (d8, m8, e8),
    (d9, m9, e9),
]


RESOURCE_AWARE_SHOTS = [
    (d1, r_m1, r_e1),
    (d1_2, r_m1_2, r_e1_2),
    (d2, r_m2, r_e2),
    (d3, r_m3, r_e3),
    (d4, r_m4, r_e4),
    (d5, r_m5, r_e5),
    (d6, r_m6, r_e6),
    (d7, r_m7, r_e7),
    (d8, r_m8, r_e8),
    (d9, r_m9, r_e9),
    (d10, r_m10, r_e10),
    # (d11, r_m11, r_e11),
]


if __name__ == "__main__":
    from pm4py import view_petri_net
    from powl import convert_to_petri_net, view

    model = m9()

    view(model)

    pn, im, fm = convert_to_petri_net(model)
    view_petri_net(pn, im, fm, format="SVG")
