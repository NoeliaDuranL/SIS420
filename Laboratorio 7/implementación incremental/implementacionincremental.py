import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Función que guarda la tabla q en un archito, 
# itera cada estado y acción y escribe el valor q correspondiente en el archivo
def save_q_table(q, filename):
    with open(filename, 'w') as f:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                f.write(f"state({i},{j}): {q[i, j, :]}\n")

# Función se utiliza para guardar las recompensas por episodio en un archivo.
def save_rewards(rewards, filename):
    with open(filename, 'w') as f:
        for episode, reward in enumerate(rewards):
            f.write(f"Episode {episode}: {reward}\n")

def run(episodes, is_training=True, render=False, epsilon=0.1):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
        visit_count = np.zeros_like(q)  # Contador de visitas para la implementación incremental
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)
        visit_count = np.zeros_like(q)

    discount_factor_g = 0.9  #El factor de descuento
    epsilon = 1 # Tasa de exploración inicial
    epsilon_decay_rate = 0.0001 # Tasa de decaimiento de la exploraicón
    rng = np.random.default_rng() #Gener número aleatorio
    # Inicializamos un arreglo para almacenar las recompensas por episodio.
    rewards_per_episode = np.zeros(episodes)
    
    #Itera cada episodio, en el cual el agente toma desiciones y actualzia la tabla q, 
    # #utiliza el valor de epsilon para la exploración, muestra la recompensa acumulada para cada episodio
    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # Se elige una acción aleatoria con probabilidad epsilon
            else:
                action = np.argmax(q[state, :]) # Se elige la acción con el valor máximo de Q para el estado actual

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                visit_count[state, action] += 1  # Incrementar el contador de visitas
                alpha = 1 / visit_count[state, action]  # Cálculo de la Tasa de aprendizaje incremental

                # Actualización de Q usando
                q[state, action] += alpha * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
                #q actualizada vamos incrementando el valor en pasos pequeños y si la recompensa es mayor o menor al valor esperado

            state = new_state
        #Decaimiento de epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    env.close()

    print('Tabla Q final:')
    print(q)

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig('taxi.png')
    plt.show()

    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(15000, is_training=True, render=False, epsilon=0.1)  # Primero entrena el modelo
    run(10, is_training=False, render=True, epsilon=0.1)  # Luego usa el modelo entrenado con renderización