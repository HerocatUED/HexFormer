import torch


    
def get_labels_start_end_time(frame_wise_labels, bg_class = ["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return torch.tensor(labels), torch.tensor(starts), torch.tensor(ends)

def levenstein(pred, label, norm = False):
    m_row = len(pred)    
    n_col = len(label)
    D = torch.zeros((m_row + 1, n_col + 1), dtype=int)
    D[:, 0] = torch.arange(m_row + 1)
    D[0, :] = torch.arange(n_col + 1)
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if label[j-1] == pred[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1, j-1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
    return score


def get_labels_start_end_time_vectorized(frame_wise_labels, bg_class=["background"]):
    change_points = torch.nonzero(torch.diff(frame_wise_labels, prepend=frame_wise_labels[:1])).flatten()
    starts = torch.cat((torch.tensor([0]), change_points))
    labels = frame_wise_labels[starts]
    ends = torch.cat((change_points, torch.tensor([len(frame_wise_labels) - 1])))
    return labels, starts, ends


def levenstein_vectorized(pred, label, norm=False):
    m_row = len(pred)
    n_col = len(label)

    # Initialize the matrix
    D = torch.zeros((m_row + 1, n_col + 1), dtype=torch.float32)

    # Initialize the first row and first column
    D[:, 0] = torch.arange(m_row + 1, dtype=torch.float32)
    D[0, :] = torch.arange(n_col + 1, dtype=torch.float32)

    # Create tensors for pred and label with padding
    pred_expanded = pred.unsqueeze(1).expand(-1, n_col)
    label_expanded = label.unsqueeze(0).expand(m_row, -1)

    # Create a match matrix where matches are 0 and mismatches are 1
    match_matrix = (pred_expanded != label_expanded).float()

    # Fill the matrix using dynamic programming
    for i in range(1, m_row + 1):
        for j in range(1, n_col + 1):
            cost = match_matrix[i - 1, j - 1]
            D[i, j] = torch.min(torch.tensor([D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost]))

    if norm:
        score = (1 - D[m_row, n_col] / max(m_row, n_col)) * 100
    else:
        score = D[m_row, n_col]

    return score


def edit_score(pred, label, norm = True, bg_class = ["background"]):
    P, _, _ = get_labels_start_end_time(pred, bg_class)
    Y, _, _ = get_labels_start_end_time(label, bg_class)
    return levenstein(P, Y, norm), levenstein_vectorized(P, Y, norm)

# Example usage
pred = torch.ones(15)
pred[1: 5] *= 6
pred[5: 8] *= 2
pred[8: 13] *= 5
pred[13: ] *= 9
label = pred.clone()
label[11: ] = 3
label[1: 4] = 7
label[4: 5] = 21
label[5: 8] = 23
print(pred)
print(label)
print(edit_score(pred, label))


labels, starts, ends = get_labels_start_end_time_vectorized(label)
labels0, starts0, ends0 = get_labels_start_end_time(label)
print("lables\n", labels, "\n", labels0)
print("starts\n", starts, "\n", starts0)
print("ends\n", ends, "\n", ends0)