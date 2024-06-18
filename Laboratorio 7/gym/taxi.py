import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    #inicializa el entorno Texi-v3
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if(is_training):
        # crea una tabla q inicializada con ceros para todas las combinaciones estado-accion
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 500 x 6 array
    else:
        # carga la talba q desde un archivo
        f = open('taxi.pkl', 'rb') #abre
        q = pickle.load(f) #carga
        f.close() #cierra

    learning_rate_a = 0.9 # tasa de aprendizaje para acturalizar la tabala q
    discount_factor_g = 0.9 # factor de descuento para las recompensas futuras
    epsilon = 1         # probabilidad inicial de exploracion (acciones aleatorias)
    epsilon_decay_rate = 0.0001        # tasa de decaimiento de epsilon para reducirla exploracion con el tiempo
    rng = np.random.default_rng()   # generador de numeros aleatorios
    
    #inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_per_episode = np.zeros(episodes)
    
    #bucle pincipal de entrenamiento
    for i in range(episodes):
        
        # Reinicial el entorno cada 1000 episodios, altenando entre modos con y sin renderizacion
        if (i * 1) % 1000 == 0:
            env.close()
            env = gym.make('Taxi-v3', render_mode = 'human')
        else:
            env.reset()
            env = gym.make('Taxi-v3')
               
        # Reinicia el entorno y establece el estado inicial
        state = env.reset()[0]
        #done = False

        # Variables para controlar la finalizacion del episodio
        terminated = False
        truncated = False
        
        rewards = 0
        # bucle para cada paso dentro de un episodio
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # exploracion:seleccion de una accion aleatorioa
            else:
                action = np.argmax(q[state,:]) # explotacion
                
            #realiza la aacion y obtine eel nuevo estado y la recompensa
            new_state,reward,terminated,truncated,_ = env.step(action)

            rewards += reward # suma la recompensa obtenida

            if is_training:
                #actualiza la tabla q con la nueva informacion obtenida
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
        #reduce epsilon para disminuir la exploracion a los largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)


        if(epsilon==0): #si epsilon es = 0, establece una nueva tasa de aprendizaje a uin valor muy pequenio
            learning_rate_a = 0.0001
            
        rewards_per_episode[i] = rewards #almacena las recompensasa acumulada
        
        if(i+1) % 50 == 0:
            print(f'episodio: {i+1} - Recompensa: {rewards_per_episode[i]}')

    env.close()
    
    # imprimir la mejor tabla Q obtenida
    print('Tabla Q final:')
    print(q)
    
    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')
    plt.show()
    
    if is_training:
        f = open("taxi.pkl","wb")
        pickle.dump(q, f)
        f.close()

    
if __name__ == '__main__':
    run(15000)    