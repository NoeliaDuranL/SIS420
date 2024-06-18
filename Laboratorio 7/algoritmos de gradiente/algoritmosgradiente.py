import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def softmax(H):

    # Calcula la exponencial de cada valor en H.
    exp_H = np.exp(H)
    # Divide cada valor exponencial por la suma de todos los valores exponenciales para obtener una distribución de probabilidad.
    return exp_H / np.sum(exp_H)

# Esta función actualiza las preferencias H basadas en la acción tomada, la recompensa obtenida, la recompensa promedio, las probabilidades y la tasa de aprendizaje.
def update_preferences(H, action, reward, average_reward, probabilities, learning_rate):
    # Actualiza la preferencia del acción tomada con la fórmula específica.
    H[action] += learning_rate * (reward - average_reward) * (1 - probabilities[action])
    for a in range(len(H)):
        if a != action:
            # Actualiza las preferencias de las acciones no tomadas.
            H[a] -= learning_rate * (reward - average_reward) * probabilities[a]
    return H

# Esta función ejecuta el entrenamiento o la evaluación del agente por un número dado de episodios.
def run(episodes, is_training=True, render=False, learning_rate=0.1):
    # Crea el entorno Taxi-v3 de Gymnasium.
    env = gym.make('Taxi-v3', render_mode='human' if render else None)
    
    if is_training:
        # Inicializa las preferencias H con ceros si estamos entrenando.
        H = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # Carga las preferencias H desde un archivo si estamos evaluando.
        with open('taxi.pkl', 'rb') as f:
            H = pickle.load(f)

    # Inicializa un array para almacenar las recompensas por episodio.
    rewards_per_episode = np.zeros(episodes)
    reward_history = []

    for i in range(episodes):
        # Reinicia el entorno y obtiene el estado inicial.
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            # Calcula las probabilidades softmax para el estado actual.
            probabilities = softmax(H[state])
            # Selecciona una acción basada en las probabilidades.
            action = np.random.choice(np.arange(env.action_space.n), p=probabilities)

            # Ejecuta la acción en el entorno.
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if is_training:
                # Calcula la recompensa promedio si hay historial de recompensas.
                average_reward = np.mean(reward_history) if reward_history else 0
                # Actualiza las preferencias H para el estado actual.
                H[state] = update_preferences(H[state], action, reward, average_reward, probabilities, learning_rate)
                # Agrega la recompensa al historial de recompensas.
                reward_history.append(reward)

            # Actualiza el estado actual.
            state = new_state

        # Guarda la recompensa total del episodio actual.
        rewards_per_episode[i] = total_reward
        if (i + 1) % 50 == 0:
            # Imprime la recompensa del episodio cada 50 episodios.
            print(f'Episodio: {i + 1} - Recompensa: {total_reward}')

    # Cierra el entorno.
    env.close()

    if is_training:
        # Guarda las preferencias H en un archivo al finalizar el entrenamiento.
        with open("taxi.pkl", "wb") as f:
            pickle.dump(H, f)

    # Grafica las recompensas por episodio.
    plt.plot(rewards_per_episode)
    plt.savefig('taxi.png')
    plt.show()

# Ejecución principal del script.
if __name__ == '__main__':
    # Ejecuta el entrenamiento del agente.
    run(15000, is_training=True, render=False, learning_rate=0.1)
    # Ejecuta la evaluación del agente.
    run(10, is_training=False, render=True, learning_rate=0.1)
