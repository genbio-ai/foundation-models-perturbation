### Purpose

These scripts calculate the differentially expressed genes. Each gene is either expressed up (+1), down (-1), not differentially expressed (0). We perform a t-test on each plate, then aggregate with voting. The final category is the one with the most votes. In the case of a tie we resolve by summing the categories that are tied and taking the sign such that (0,1 -> 1) (0,-1 -> -1) (1,-1 -> 0) and (1,0,-1 -> 0). We remove batches for which a drug is present in less than 20 cells.
