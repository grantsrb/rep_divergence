import torch

def make_tasks(task_division="distinct_y", varbs=None):
    """
    This function splits the classes into two different tasks.
    
    Args:
        task_division: str
            argue the type of task division
            options:
                distinct_y: task 1 and task 2 have 4 distinct y values with
                    one example having a unique x value from the other 3 classes.
                    Task one and task 2 have no overlapping examples.
                distinct_y_overlap: task 1 and task 2 have 4 distinct y values with
                    one example having a unique x value from the other 3 classes.
                    Task one and task 2 have 2 overlapping examples.
                distinct_xy: classes in task 1 and task 2 have distinct x and y
                    values. None of the classes overlap between the two tasks.
                distinct_xy_overlap: classes in task 1 and task 2 have distinct x
                    and y values. Some of the classes overlap between the two tasks.
                shared_y: task 1 and task 2 have 4 y values of which one is
                    shared between two separate classes. None of the classes
                    overlap between the two tasks.
                shared_y_overlap: task 1 and task 2 have 4 y values of which one is
                    shared between two separate classes. Some of the classes
                    overlap between the two tasks.
                tetris_L: task 1 and task 2 have 4 y values of which the top
                    is shared between two separate classes for one task
                    and the bottom is shared between two separate classes for
                    the other task. The odd-one-out x value overlaps between
                    the two tasks.
                random: classes are split randomly between the two tasks. No
                    overlap between the two tasks.
                random_overlap: classes are split randomly between the two tasks.
                    Some of the classes overlap between the two tasks.
        varbs: ndarray or tensor
            The variables to split into tasks. Assumes a two dimensional
            tensor with x,y values in each row. Further, assumes x can
            be -1 or 1, and y can be 0,1,2, 3, or 4.
    """
    if task_division=="distinct_y":
        task1_tups = {
                    (1,0),
                    (1,1),
                    (1,2),
                    (1,3),
            (-1,4)
        }
        task2_tups = {
                    (1,4),
            (-1,0),
            (-1,1),
            (-1,2),
            (-1,3),
        }
    elif task_division=="distinct_y_overlap":
        task1_tups = {
                    (1,0),
            (-1,1),
                    (1,2),
                    (1,3),
            (-1,4),
        }
        task2_tups = {
            (-1,0),
            (-1,1),
                    (1,2),
            (-1,2),
            (-1,3),
                    (1,4),
        }
    elif task_division=="distinct_xy":
        task1_tups = {
                    (1,0),      
            (-1,1), 
        }
        task2_tups = {


            (-1,2), 
                    (1,3),      
        }
    elif task_division=="distinct_xy_overlap":
        task1_tups = {
                    (1,0),      
            (-1,1), 
        }
        task2_tups = {

            (-1,1), 
                    (1,2),      
        }
    elif task_division=="xor":
        task1_tups = {
                    (1,0),      
            (-1,1), 
        }
        task2_tups = {
            (-1,0), 
                   (1,1),      
        }
    elif task_division=="og_noholdouts":
        task1_tups = {
            (-1,0), (1,0),
            (-1,1), (1,1),
            (-1,2), (1,2),
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
        task2_tups = {
            (-1,0), (1,0),
            (-1,1), (1,1),
            (-1,2), (1,2),
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
    elif task_division=="og_holdouts" or task_division=="original":
        task1_tups = {
            (-1,0), (1,0),
            
            (-1,2), (1,2),
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
        task2_tups = {
            (-1,0), (1,0),
            (-1,1), (1,1),
            
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
    elif task_division=="shared_y":
        task1_tups = {
            (-1,0), (1,0),

            (-1,2), (1,2),
                    
            (-1,4), (1,4),
        }
        task2_tups = {

            (-1,1), (1,1),

            (-1,3), (1,3),
        }
    elif task_division=="shared_y_overlap":
        task1_tups = {
            (-1,0), (1,0),

            (-1,2), (1,2),
            (-1,3), (1,3),
        }
        task2_tups = {

            (-1,1), (1,1),
            (-1,2), (1,2),

            (-1,4), (1,4),
        }
    elif task_division=="tetris_L":
        task1_tups = {
            (-1,0), (1,0),
                    (1,1),
                    (1,2),
                    (1,3),
                    (1,4),
        }
        task2_tups = {
            (-1,0),
            (-1,1),
            (-1,2),
            (-1,3),
            (-1,4), (1,4),
        }
    elif task_division=="mirror_L":
        task1_tups = {
            (-1,0), (1,0),
                    (1,1),
                    (1,2),
                    (1,3),
                    (1,4),
        }
        task2_tups = {
            (-1,0), (1,0),
            (-1,1),
            (-1,2),
            (-1,3),
            (-1,4),
        }
    elif task_division=="tetris_T":
        task1_tups = {
                    (1,0),
                    (1,1),
            (-1,2), (1,2),
                    (1,3),
                    (1,4),
        }
        task2_tups = {
            (-1,0),
            (-1,1),
            (-1,2), (1,2),
            (-1,3),
            (-1,4),
        }
    elif task_division=="tetris_T_no_overlap":
        task1_tups = {
                    (1,0),
                    
            (-1,2), (1,2),
                    
                    (1,4),
        }
        task2_tups = {

            (-1,1),
            (-1,2), (1,2),
            (-1,3),
        }
    elif task_division=="tetris_C":
        task1_tups = {
            (-1,0), (1,0),
                    (1,1),
                    (1,2),
                    (1,3),
            (-1,4), (1,4),
        }
        task2_tups = {
            (-1,0), (1,0),
            (-1,1),
            (-1,2),
            (-1,3),
            (-1,4), (1,4),
        }
    elif task_division=="tetris_h":
        task1_tups = {
            (-1,0),
            (-1,1),
            (-1,2), (1,2),
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
        task2_tups = {
            (-1,0), (1,0),
            (-1,1), (1,1),
            (-1,2), (1,2),
                    (1,3),
                    (1,4),
        }
    elif task_division=="mirror_h":
        task1_tups = {
            (-1,0),
            (-1,1),
            (-1,2), (1,2),
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
        task2_tups = {
                    (1,0),
                    (1,1),
            (-1,2), (1,2),
            (-1,3), (1,3),
            (-1,4), (1,4),
        }
    elif task_division=="tetris_F":
        task1_tups = {
            (-1,0), (1,0),
            (-1,1),
            (-1,2), (1,2),
            (-1,3),
            (-1,4),
        }
        task2_tups = {
                    (1,0),
                    (1,1),
            (-1,2), (1,2),
                    (1,3),
            (-1,4), (1,4),
        }
    elif task_division=="mirror_F":
        task1_tups = {
            (-1,0), (1,0),
            (-1,1),
            (-1,2), (1,2),
            (-1,3),
            (-1,4),
        }
        task2_tups = {
            (-1,0), (1,0),
                    (1,1),
            (-1,2), (1,2),
                    (1,3),
                    (1,4),
        }
    elif task_division=="inner_square":
        task1_tups = {
            (-1,0), (1,0),
            
            
            (-1,3), (1,3),
        }
        task2_tups = {
            
            (-1,1), (1,1),
            (-1,2), (1,2),
        }
    elif task_division=="random":
        all_tups = {(x,y) for x in [-1,1] for y in range(5)}
        task1_tups = {
            tup for i,tup in enumerate(all_tups)
                if i in torch.randperm(len(all_tups))[:len(all_tups)//2]
        }
        task2_tups = {tup for i,tup in enumerate(all_tups) if tup not in task1_tups}
    elif task_division=="random_overlap":
        all_tups = {(x,y) for x in [-1,1] for y in range(5)}
        task1_tups = {tup for i,tup in enumerate(all_tups) if i in torch.randperm(len(all_tups))[:len(all_tups)//2]}
        task2_tups = {tup for i,tup in enumerate(all_tups) if i in torch.randperm(len(all_tups))[len(all_tups)//2:]}
    else:
        raise ValueError(f"Invalid task division: {task_division}")
    task1_bools = torch.tensor([(int(x),int(y)) in task1_tups for x,y in varbs])
    task2_bools = torch.tensor([(int(x),int(y)) in task2_tups for x,y in varbs])
    return {"bools": [task1_bools, task2_bools], "tups": [task1_tups, task2_tups]}

all_divisions = [
    "og_noholdouts",
    "og_holdouts",
    "shared_y",
    "shared_y_overlap",
    "tetris_L",
    "mirror_L",
    "tetris_T",
    "tetris_T_no_overlap",
    "tetris_C",
    "tetris_h",
    "mirror_h",
    "tetris_F",
    "mirror_F",
    "random",
    "random_overlap",
    "inner_square",
    "xor",
    "distinct_xy_overlap",
    "distinct_xy",
    "distinct_y_overlap",
    "distinct_y",
]