import numpy as np


# Hàm tính lực hấp dẫn (Attractive Force)
def attractive_force(Va, d_att, alpha_a):
    """
    Tính lực hấp dẫn Fatt

    Va: Lực hấp dẫn tối đa
    d_att: Vector chỉ hướng từ Q_near đến Q_end (dạng numpy array)
    alpha_a: Hệ số điều chỉnh

    Trả về: Vector lực hấp dẫn Fatt
    """
    d_att_norm = np.linalg.norm(d_att)  # Độ lớn của vector d_att
    if d_att_norm == 0:
        return np.zeros_like(d_att)  # Tránh chia cho 0

    Fatt = Va * (1 - np.exp(-d_att_norm ** alpha_a)) * (d_att / d_att_norm)
    return Fatt


# Hàm tính lực đẩy (Repulsive Force)
def repulsive_force(Vr, d_rep, d_safe, alpha_r):
    """
    Tính lực đẩy Frep

    Vr: Lực đẩy tối đa
    d_rep: Vector chỉ hướng từ robot đến vật cản (dạng numpy array)
    d_safe: Khoảng cách an toàn
    alpha_r: Hệ số điều chỉnh

    Trả về: Vector lực đẩy Frep
    """
    d_rep_norm = np.linalg.norm(d_rep)  # Độ lớn của vector d_rep
    if d_rep_norm == 0:
        return np.zeros_like(d_rep)  # Tránh chia cho 0

    Frep = (d_rep / d_rep_norm) * (Vr / (1 + np.exp(d_rep_norm - d_safe ** alpha_r)))
    return Frep


# Hàm thử nghiệm
def main():
    # Thử nghiệm hàm lực hấp dẫn
    Q_near = np.array([0, 0, 0, 0, 0, 0])  # Góc robot Q_near
    Q_end = np.array([1, 1, 1, 1, 1, 1])  # Góc robot Q_end
    d_att = Q_end - Q_near  # Vector hướng từ Q_near đến Q_end
    Va = 10  # Lực hấp dẫn tối đa
    alpha_a = 2  # Hệ số điều chỉnh

    Fatt = attractive_force(Va, d_att, alpha_a)
    print("Lực hấp dẫn Fatt:", Fatt)

    # Thử nghiệm hàm lực đẩy
    d_rep = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Vector từ robot đến vật cản
    Vr = 20  # Lực đẩy tối đa
    d_safe = 1  # Khoảng cách an toàn
    alpha_r = 2  # Hệ số điều chỉnh

    Frep = repulsive_force(Vr, d_rep, d_safe, alpha_r)
    print("Lực đẩy Frep:", Frep)


if __name__ == "__main__":
    main()
