import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qutip import Bloch, basis, Qobj, tensor
import random

# Set ffmpeg path if necessary
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# ====== PART 1: DEUTSCH-JOZSA ALGORITHM ======

def hadamard_transform(n):
    H = Qobj([[1, 1], [1, -1]], dims=[[2], [2]]) / np.sqrt(2)
    return tensor([H] * n)

def oracle_gate(n_qubits, f_type="balanced"):
    N = 2 ** n_qubits
    U = np.eye(N, dtype=complex)
    if f_type == "balanced":
        for i in range(N):
            if bin(i).count('1') % 2 == 1:
                U[i, i] *= -1
    U_op = Qobj(U)
    U_op.dims = [[2] * n_qubits, [2] * n_qubits]
    return U_op

def interpolate_states(start, end, num_steps):
    states = []
    for i in range(num_steps):
        ratio = i / num_steps
        psi = (1 - ratio) * start + ratio * end
        if psi.norm() < 1e-12:
            psi = psi + 1e-6 * start
        psi = psi.unit()
        states.append(psi)
    return states

def run_deutsch_jozsa_animation(states_list, title="Deutsch-Jozsa", filename="deutsch_jozsa_2qubit.mp4", view=(60, 30), fps=15):
    fig = plt.figure(figsize=(12, 6))
    ax_bloch = fig.add_subplot(121, projection='3d')
    ax_bar = fig.add_subplot(122)

    bloch = Bloch(fig=fig, axes=ax_bloch)
    bloch.vector_color = ['b']
    bloch.point_color = ['b']
    bloch.point_marker = ['o']
    bloch.point_size = [20]
    bloch.show_axes_label = True
    bloch.frame_alpha = 0.1

    def update(i):
        bloch.clear()
        state_one_qubit = states_list[i].ptrace(0)
        bloch.add_states(state_one_qubit)
        bloch.make_sphere()
        bloch.set_label_convention("xyz")
        bloch.xlabel = ['$x$', '']
        bloch.ylabel = ['$y$', '']
        bloch.zlabel = ['$z$', '']
        bloch.fig.suptitle(f"{title} - Frame {i}", y=0.95)
        bloch.axes.view_init(view[0], view[1])

        ax_bar.cla()
        probs = np.abs(states_list[i].full())**2
        x = np.arange(len(probs))
        ax_bar.bar(x, probs.flatten(), color=['C' + str(i % 10) for i in range(len(probs))])
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title("Measurement Probabilities")
        ax_bar.set_ylabel("Probability")
        ax_bar.set_xlabel("Basis State")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"|{i:02b}>" for i in range(len(probs))], rotation=45)

    ani = FuncAnimation(fig, update, len(states_list), interval=100, repeat=False)
    try:
        ani.save(filename, fps=fps, dpi=200, writer="ffmpeg")
        print(f"\n✅ Saved animation to {filename}")
    except Exception as e:
        print(f"\n❌ Error saving animation: {e}")
    plt.close(fig)

def plot_frame(state, frame_number, filename_prefix="deutsch_jozsa_frame", view=(60, 30)):
    fig = plt.figure(figsize=(12, 6))
    ax_bloch = fig.add_subplot(121, projection='3d')
    ax_bar = fig.add_subplot(122)

    bloch = Bloch(fig=fig, axes=ax_bloch)
    bloch.vector_color = ['b']
    bloch.point_color = ['b']
    bloch.point_marker = ['o']
    bloch.point_size = [20]
    bloch.show_axes_label = True
    bloch.frame_alpha = 0.1

    state_one_qubit = state.ptrace(0)
    bloch.add_states(state_one_qubit)
    bloch.make_sphere()
    bloch.set_label_convention("xyz")
    bloch.xlabel = ['$x$', '']
    bloch.ylabel = ['$y$', '']
    bloch.zlabel = ['$z$', '']
    bloch.fig.suptitle(f"Quantum State Evolution - Frame {frame_number}", y=0.95)
    bloch.axes.view_init(view[0], view[1])

    N = len(state.full().flatten())
    x = np.arange(N)
    probs = np.abs(state.full())**2
    bar_colors = ['C' + str(i % 10) for i in range(N)]

    ax_bar.bar(x, probs.flatten(), color=bar_colors)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title("Measurement Probabilities")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xlabel("Basis State")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"|{i:02b}>" for i in range(N)], rotation=45)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{frame_number}.png", dpi=200)
    plt.close(fig)

def main_deutsch_jozsa():
    print("\nRunning Deutsch-Jozsa Algorithm (2-qubit version)")
    n_qubits = 2
    psi0 = tensor([basis(2, 0), basis(2, 1)])
    H = hadamard_transform(n_qubits)
    psi_plus = H * psi0
    oracle = oracle_gate(n_qubits, f_type="balanced")
    psi_marked = oracle * psi_plus
    psi_final = H * psi_marked

    num_transition_steps = 50
    states_list = []
    states_list += interpolate_states(psi0, psi_plus, num_transition_steps)
    states_list += interpolate_states(psi_plus, psi_marked, num_transition_steps)
    states_list += interpolate_states(psi_marked, psi_final, num_transition_steps)

    run_deutsch_jozsa_animation(states_list)

    for frame in [0, 49, 76, 100, 149]:
        if frame < len(states_list):
            plot_frame(states_list[frame], frame)

# ====== PART 2: GROVER SEARCH ======

def generate_index_to_value_map(MAX_NUMBER=1024):
    values = list(range(MAX_NUMBER))
    rng = random.Random(42)
    rng.shuffle(values)
    return values

MAX_NUMBER = 1024
index_to_value = generate_index_to_value_map(MAX_NUMBER)

def pseudo_shuffle(index, MAX_NUMBER=1024):
    return index_to_value[index]

def find_address_in_table(target_number, MAX_NUMBER=1024):
    for index in range(MAX_NUMBER):
        if pseudo_shuffle(index, MAX_NUMBER) == target_number:
            return index
    return None

def grover_oracle(n_qubits, marked_value, MAX_NUMBER=4):
    N = 2 ** n_qubits
    U = np.eye(N, dtype=complex)
    for i in range(N):
        if i == marked_value:
            U[i, i] *= -1
    return Qobj(U, dims=[[2]*n_qubits, [2]*n_qubits])

def diffusion_operator(n_qubits):
    N = 2 ** n_qubits
    uniform = np.ones((N, N)) / N
    I = np.eye(N)
    return Qobj(2 * uniform - I, dims=[[2]*n_qubits, [2]*n_qubits])

def run_grover_search_with_animation(n_qubits, target_number, MAX_NUMBER=4):
    psi = tensor([basis(2, 0)] * n_qubits)
    H = hadamard_transform(n_qubits)
    psi = H * psi

    oracle = grover_oracle(n_qubits, target_number, MAX_NUMBER)
    diffusion = diffusion_operator(n_qubits)

    states = [psi.ptrace(0)]
    iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** n_qubits)))

    for _ in range(iterations):
        psi = oracle * psi
        psi = diffusion * psi
        states.append(psi.ptrace(0))

    animate_bloch(states)

def animate_bloch(states, filename="grover_animation.mp4", fps=5, view=(60, 30), num_interpolation_frames=10):
    if not states or len(states) < 2:
        print("Not enough states to animate.")
        return

    fig = plt.figure(figsize=(12, 6))
    ax_bloch = fig.add_subplot(121, projection='3d')
    ax_bar = fig.add_subplot(122)

    bloch = Bloch(fig=fig, axes=ax_bloch)
    bloch.vector_color = ['b']
    bloch.point_color = ['b']
    bloch.point_marker = ['o']
    bloch.point_size = [20]
    bloch.show_axes_label = True
    bloch.frame_alpha = 0.1

    interpolated_states = []
    for i in range(len(states) - 1):
        start = states[i]
        end = states[i + 1]
        for j in range(num_interpolation_frames):
            ratio = j / num_interpolation_frames
            interp = (1 - ratio) * start + ratio * end
            interp = interp.unit()
            interpolated_states.append(interp)

    def update(i):
        bloch.clear()
        bloch.add_states(interpolated_states[i])
        bloch.make_sphere()
        bloch.set_label_convention("xyz")
        bloch.xlabel = ['$x$', '']
        bloch.ylabel = ['$y$', '']
        bloch.zlabel = ['$z$', '']
        bloch.fig.suptitle(f"Grover Iteration {i // num_interpolation_frames}", y=0.95)
        bloch.axes.view_init(view[0], view[1])

        ax_bar.cla()
        probs = np.abs(interpolated_states[i].full())**2
        num_probs = len(probs.flatten())
        x = np.arange(num_probs)
        colors = [f'C{k}' for k in range(num_probs)]
        ax_bar.bar(x, probs.flatten(), color=colors)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title("Measurement Probabilities")
        ax_bar.set_ylabel("Probability")
        ax_bar.set_xlabel("Basis State")

    ani = FuncAnimation(fig, update, len(interpolated_states), interval=200, repeat=False)
    try:
        ani.save(filename, fps=fps, dpi=200, writer="ffmpeg")
        print(f"✅ Saved Grover animation to {filename}")
    except Exception as e:
        print(f"❌ Error saving animation: {e}")
    plt.close(fig)

# ====== MAIN MENU ======

def main():
    while True:
        print("\n=== Quantum Algorithm Demo ===")
        print("1. Run Deutsch-Jozsa Animation")
        print("2. Run Grover Search with Animation")
        print("3. Exit")
        choice = input("Enter option (1/2/3): ").strip()

        if choice == '1':
            print("\nStarting Deutsch-Jozsa Animation...")
            main_deutsch_jozsa()
        elif choice == '2':
            print(f"\nTable contains shuffled numbers from 0 to {MAX_NUMBER - 1}")
            user_input = input(f"Enter a number to search (0–{MAX_NUMBER - 1}): ").strip()
            if not user_input.isdigit():
                print("Invalid input. Please enter a valid number.")
                continue
            target_number = int(user_input)
            if not (0 <= target_number < MAX_NUMBER):
                print(f"Number out of range. Must be between 0 and {MAX_NUMBER - 1}.")
                continue
            address = find_address_in_table(target_number)
            if address is not None:
                print(f"\nNumber {target_number} found at address {address} in the table.")
            else:
                print(f"\nNumber {target_number} not found in the table.")
                continue
            print("\nNow running Grover search animation...\n")
            run_grover_search_with_animation(n_qubits=2, target_number=target_number % 4)
        elif choice == '3':
            print("\nExiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
