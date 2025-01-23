
def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j - 1]:
                count += 1
            else:
                encoded_row.append(f"{row[j - 1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append("| ".join(encoded_row))
    return "\n ".join(rows) + "\n"