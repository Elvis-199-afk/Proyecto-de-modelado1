from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
from scipy.optimize import fsolve # Para la función de Newton-Raphson (opcional)
import matplotlib.pyplot as plt
import io
import base64
# Usar el backend 'Agg' para matplotlib
import matplotlib
matplotlib.use('Agg') 

# ==========================================================
# CONFIGURACIÓN INICIAL
# ==========================================================
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir peticiones desde el frontend

# Diccionario de funciones matemáticas disponibles para eval
MATH_FUNCTIONS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
    'abs': np.abs, 'np': np # Se expone todo numpy como 'np' y sus funciones principales
}

# Función auxiliar para convertir listas anidadas de strings a matrices NumPy
def parse_matrix(matrix_list):
    try:
        # Intenta convertir los valores a float
        return np.array([list(map(float, row)) for row in matrix_list])
    except Exception as e:
        raise ValueError(f"Error al parsear la matriz. Asegúrate de que todos los valores sean números. Detalle: {e}")

# Auxiliar: Parsea el string de la función (ej: "x**2 - 4")
def parse_function(func_str, variable='x'):
    code = f'lambda {variable}: {func_str}'
    try:
        # Hacemos disponibles las funciones matemáticas para el eval
        context = {**MATH_FUNCTIONS}
        return eval(code, context, {}) 
    except Exception:
        raise ValueError(f"Función inválida. Asegúrate de usar '{variable}' como variable y operadores Python válidos (e.g., 'cos', 'np.sin').")

# Auxiliar: Parsea un vector de funciones F(x) o G(x) donde x es un vector (x1, x2, ...)
def parse_vector_functions(func_list, N):
    if N == 2:
        var_names = 'x1, x2'
    elif N == 3:
        var_names = 'x1, x2, x3'
    else:
        raise ValueError("Número de variables no soportado.")
        
    compiled_funcs = []
    context = {**MATH_FUNCTIONS}
    for i, func_str in enumerate(func_list):
        code = f'lambda {var_names}: {func_str}'
        try:
            compiled_funcs.append(eval(code, context, {}))
        except Exception:
            raise ValueError(f"Función F/G {i+1} inválida. Asegúrate de usar las variables correctas ({var_names}) y funciones Python (e.g., 'cos', 'np.sin').")
            
    return lambda x: np.array([f(*x) for f in compiled_funcs])


# ==========================================================
# 1. OPERACIONES BÁSICAS DE MATRICES (Suma, Mult, Det, Inv)
# ==========================================================
@app.route('/api/matrix', methods=['POST'])
def handle_matrix_operation():
    data = request.json
    operation = data.get('operation')
    matrix_a_list = data.get('matrixA')
    matrix_b_list = data.get('matrixB', None)

    try:
        A = parse_matrix(matrix_a_list)
        
        # Determinante e Inversa (Requieren 1 matriz cuadrada)
        if operation in ['determinant', 'inverse']:
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return jsonify({"error": "La matriz debe ser cuadrada (N x N)."}), 400

            if operation == 'determinant':
                result = np.linalg.det(A)
                return jsonify({"result": float(result)})

            elif operation == 'inverse':
                det_val = np.linalg.det(A)
                if np.isclose(det_val, 0):
                    return jsonify({"error": "La matriz es singular (Determinante ≈ 0) y no tiene inversa."}), 400
                result = np.linalg.inv(A).tolist()
                return jsonify({"result": result})

        # Suma y Multiplicación (Requieren 2 matrices)
        elif operation in ['sum', 'multiply']:
            B = parse_matrix(matrix_b_list)

            if operation == 'sum':
                if A.shape != B.shape:
                    return jsonify({"error": "Las matrices deben tener las mismas dimensiones para la Suma."}), 400
                result = (A + B).tolist()
                return jsonify({"result": result})

            elif operation == 'multiply':
                # Condición de multiplicación: columnas de A = filas de B
                if A.shape[1] != B.shape[0]:
                    return jsonify({"error": f"Dimensiones incompatibles para la Multiplicación. Col. A ({A.shape[1]}) != Fil. B ({B.shape[0]})"}), 400
                result = np.dot(A, B).tolist()
                return jsonify({"result": result})
        
        return jsonify({"error": "Operación de matriz no válida."}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error inesperado en la operación de matriz: {e}"}), 500

# ==========================================================
# 2. MÉTODOS DE SISTEMAS DE ECUACIONES (Gauss, Jordan, Iterativos)
# ==========================================================

# Auxiliar para inicializar la solución y la matriz T, c para iterativos
def setup_iterative(A, b):
    n = len(b)
    x0 = np.zeros(n)
    T = np.zeros((n, n), dtype=float)
    c = np.zeros(n, dtype=float)
    
    for i in range(n):
        if np.isclose(A[i, i], 0):
             # Esto debería ser manejado por pivoteo para métodos directos
             # Para iterativos, si el pivote es 0, el método falla o hay que reordenar
             raise ValueError(f"División por cero. El elemento diagonal A[{i},{i}] es cero. Intente reordenar las ecuaciones.")
        
        for j in range(n):
            if i != j:
                T[i, j] = -A[i, j] / A[i, i]
        c[i] = b[i] / A[i, i]
        
    return x0, T, c

# --- Funciones de Métodos ---

def gauss_elimination(A_orig, b_orig):
    A = A_orig.copy()
    b = b_orig.copy()
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1)])
    
    for k in range(n):
        # Pivoteo parcial
        max_row = np.argmax(abs(M[k:,k])) + k
        M[[k, max_row]] = M[[max_row, k]]
        
        if np.isclose(M[k][k], 0):
            raise ValueError("El sistema no tiene solución única (pivote cero después de pivoteo).")

        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            M[i] = M[i] - factor * M[k]
            
    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # np.dot(M[i,i+1:n], x[i+1:n]) calcula la suma de los términos ya conocidos
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]
    return x.tolist()

def gauss_jordan(A_orig, b_orig):
    A = A_orig.copy()
    b = b_orig.copy()
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1)])
    
    for k in range(n):
        # Pivoteo parcial
        max_row = np.argmax(abs(M[k:,k])) + k
        M[[k, max_row]] = M[[max_row, k]]
        
        if np.isclose(M[k][k], 0):
            raise ValueError("El sistema no tiene solución única (pivote cero después de pivoteo).")
            
        # Normalización (hace que M[k][k] sea 1)
        M[k] = M[k] / M[k][k]
        
        # Eliminación
        for i in range(n):
            if i != k:
                M[i] = M[i] - M[i][k] * M[k]
                
    return M[:, -1].tolist()

def jacobi(A_orig, b_orig, tol=1e-6, max_iter=100):
    A = A_orig.copy()
    b = b_orig.copy()
    x0, T, c = setup_iterative(A, b)
    x = x0.copy()
    
    for k in range(max_iter):
        x_new = np.dot(T, x) + c
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new.tolist(), k + 1
            
        x = x_new
        
    raise Exception(f"El método de Jacobi no convergió después de {max_iter} iteraciones (Tolerancia: {tol}).")

def gauss_seidel(A_orig, b_orig, tol=1e-6, max_iter=100):
    A = A_orig.copy()
    b = b_orig.copy()
    n = len(b)
    x = np.zeros(n)
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            # Suma de términos con valores nuevos de x (ya calculados en esta iteración)
            sum_new = np.dot(A[i, :i], x[:i]) 
            # Suma de términos con valores antiguos de x
            sum_old = np.dot(A[i, i+1:], x_old[i+1:]) 
            
            x[i] = (b[i] - sum_new - sum_old) / A[i, i]
        
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x.tolist(), k + 1
            
    raise Exception(f"El método de Gauss-Seidel no convergió después de {max_iter} iteraciones (Tolerancia: {tol}).")

# --- Ruta Principal de Sistemas ---

@app.route('/api/system', methods=['POST'])
def solve_system():
    data = request.json
    method = data.get('method')
    matrix_a_list = data.get('matrixA')
    vector_b_list = data.get('vectorB')
    
    # Parámetros para métodos iterativos
    max_iter = data.get('maxIter', 100)
    tol = data.get('tolerance', 1e-6)

    try:
        A = parse_matrix(matrix_a_list)
        b = np.array(vector_b_list, dtype=float)

        if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
            return jsonify({"error": "La matriz de coeficientes debe ser cuadrada (N x N) y el vector b debe tener N elementos."}), 400

        iterations = None
        
        if method == 'gauss_elimination':
            result_vector = gauss_elimination(A, b)
        elif method == 'gauss_jordan':
            result_vector = gauss_jordan(A, b)
        elif method == 'jacobi':
            result_vector, iterations = jacobi(A, b, tol, max_iter)
        elif method == 'gauss_seidel':
            result_vector, iterations = gauss_seidel(A, b, tol, max_iter)
        else:
            return jsonify({"error": "Método de solución de sistema no válido."}), 400
            
        formatted_result = {f"x{i+1}": val for i, val in enumerate(result_vector)}
        
        return jsonify({
            "result": formatted_result, 
            "iterations": iterations 
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error en la ejecución del método: {e}"}), 500



# ==========================================================
# 3. MÉTODOS DE BÚSQUEDA DE RAÍCES
# ==========================================================

# Auxiliar para calcular la Jacobiana (usada por Newton normal y modificado)
def jacobian(F, x, N, h=1e-6):
    J = np.zeros((N, N))
    x_h = x.copy()
    
    for j in range(N):
        x_h[j] = x[j] + h
        Fx_plus_h = F(x_h)
        
        x_h[j] = x[j] - h
        Fx_minus_h = F(x_h)
        
        J[:, j] = (Fx_plus_h - Fx_minus_h) / (2 * h)
        x_h[j] = x[j]
        
    return J

# ----------------------------------------------------------
# BISECCIÓN (N=1) - F(x)=0
# ----------------------------------------------------------
def bisection(func_str, a, b, tol, max_iter):
    f = parse_function(func_str, variable='x')
    
    if f(a) * f(b) > 0:
        raise ValueError("El método de Bisección requiere que f(a) y f(b) tengan signos opuestos.")
    
    if a > b: a, b = b, a

    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, k
            
        if f(a) * fc < 0:
            b = c
        else:
            a = c
            
    raise Exception(f"Bisección no convergió después de {max_iter} iteraciones.")

# ----------------------------------------------------------
# SECANTE (N=1) - F(x)=0
# ----------------------------------------------------------
def secant_method(func_str, x_prev, x_curr, tol, max_iter):
    f = parse_function(func_str, variable='x')
    
    x_prev = float(x_prev)
    x_curr = float(x_curr)
    
    for k in range(1, max_iter + 1):
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        
        if np.isclose(f_curr, f_prev):
            raise Exception("División por cero. El cambio en f(x) es demasiado pequeño.")
            
        x_next = x_curr - f_curr * ((x_curr - x_prev) / (f_curr - f_prev))
        
        if abs(x_next - x_curr) < tol or abs(f(x_next)) < tol:
            return x_next, k
            
        x_prev = x_curr
        x_curr = x_next
        
    raise Exception(f"Secante no convergió después de {max_iter} iteraciones.")

# ----------------------------------------------------------
# PUNTO FIJO PARA SISTEMAS (N=2, 3) - G(x)=x
# ----------------------------------------------------------
def fixed_point_system(func_list, x0, N, tol=1e-6, max_iter=100):
    G = parse_vector_functions(func_list, N)
    x = np.array(x0, dtype=float)

    for k in range(1, max_iter + 1):
        x_new = G(x)
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new.tolist(), k
            
        x = x_new
        
    raise Exception(f"Punto Fijo no convergió después de {max_iter} iteraciones.")

# ----------------------------------------------------------
# NEWTON-RAPHSON (N=2, 3) - F(x)=0
# ----------------------------------------------------------
def newton_raphson_system(func_list, x0, N, tol=1e-6, max_iter=100):
    F = parse_vector_functions(func_list, N)
    x = np.array(x0, dtype=float)

    for k in range(1, max_iter + 1):
        Fx = F(x)
        
        if np.linalg.norm(Fx, ord=np.inf) < tol:
            return x.tolist(), k - 1 

        J = jacobian(F, x, N)
        
        if np.linalg.det(J) == 0:
            raise Exception("La matriz Jacobiana es singular en la iteración, el método falló.")

        p = np.linalg.solve(J, -Fx)
        x_new = x + p
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new.tolist(), k
            
        x = x_new
        
    raise Exception(f"Newton-Raphson no convergió después de {max_iter} iteraciones.")

# ----------------------------------------------------------
# NEWTON RAPHSON MODIFICADO (N=2, 3) - F(x)=0
# ----------------------------------------------------------
def jacobian_frozen_newton(func_list, x0, N, tol=1e-6, max_iter=100, freeze_iter=5):
    """Resuelve F(x) = 0 usando Newton, actualizando la Jacobiana cada 'freeze_iter' veces."""
    F = parse_vector_functions(func_list, N)
    x = np.array(x0, dtype=float)
    J_inv = None 
    
    for k in range(1, max_iter + 1):
        Fx = F(x)
        
        if np.linalg.norm(Fx, ord=np.inf) < tol:
            return x.tolist(), k - 1 

        if J_inv is None or (k - 1) % freeze_iter == 0:
            J = jacobian(F, x, N)
            if np.linalg.det(J) == 0:
                raise Exception(f"La matriz Jacobiana es singular en la iteración {k}, el método falló.")
            J_inv = np.linalg.inv(J)
            
        p = np.dot(J_inv, -Fx)
        x_new = x + p
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new.tolist(), k
            
        x = x_new
        
    raise Exception(f"Newton Modificado no convergió después de {max_iter} iteraciones.")


# ----------------------------------------------------------
# GENERACIÓN DE GRÁFICAS (N=2)
# ----------------------------------------------------------

def generate_plot_newton(func_list, root, x_min, x_max, y_min, y_max):
    """Gráfica para F(x)=0 (Newton-Raphson)"""
    # ... (El código de esta función es idéntico al de generate_plot_2d_system anterior) ...
    f1 = parse_function(func_list[0], 'x1, x2')
    f2 = parse_function(func_list[1], 'x1, x2')
    
    x1_range = np.linspace(x_min, x_max, 200)
    x2_range = np.linspace(y_min, y_max, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    Z1 = f1(X1, X2)
    Z2 = f2(X1, X2)
    
    plt.figure(figsize=(8, 6))
    
    plt.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2, linestyles='solid', label=f'F1=0: {func_list[0]}')
    plt.contour(X1, X2, Z2, levels=[0], colors='red', linewidths=2, linestyles='dashed', label=f'F2=0: {func_list[1]}')
    
    plt.plot(root[0], root[1], 'ko', markersize=8, label=f'Raíz: ({root[0]:.4f}, {root[1]:.4f})')
    
    plt.title('Intersección del Sistema F(x) = 0 (Newton-Raphson)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def generate_plot_fixed_point(func_list, root, x_min, x_max, y_min, y_max):
    """Gráfica para G(x)=x (Punto Fijo)"""
    G1 = parse_function(func_list[0], 'x1, x2')
    G2 = parse_function(func_list[1], 'x1, x2')
    
    x1_range = np.linspace(x_min, x_max, 200)
    x2_range = np.linspace(y_min, y_max, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    plt.figure(figsize=(8, 6))
    
    # Curvas de nivel donde x1 = G1(x1, x2) => G1(x1, x2) - x1 = 0
    Z1 = G1(X1, X2) - X1
    # Curvas de nivel donde x2 = G2(x1, x2) => G2(x1, x2) - x2 = 0
    Z2 = G2(X1, X2) - X2
    
    plt.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2, linestyles='solid', label=f'$x_1=G_1(x)$')
    plt.contour(X1, X2, Z2, levels=[0], colors='red', linewidths=2, linestyles='dashed', label=f'$x_2=G_2(x)$')
    
    plt.plot(root[0], root[1], 'ko', markersize=8, label=f'Punto Fijo: ({root[0]:.6f}, {root[1]:.6f})') 
    
    plt.title('Punto Fijo de Sistemas (Intersección de G1 y G2)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


# --- Ruta Principal de Búsqueda de Raíces (COMPLETA) ---

@app.route('/api/roots', methods=['POST'])
def find_root():
    data = request.json
    method = data.get('method')
    func_list = data.get('functions') 
    N = data.get('N', 1)              
    
    # Parámetros comunes
    tol = data.get('tolerance', 1e-6)
    max_iter = data.get('maxIter', 100)
    
    try:
        if len(func_list) != N:
            raise ValueError(f"Debe proporcionar {N} funciones para un sistema de {N} variables.")
            
        iterations = None
        image_data = None
        
        # --- MÉTODOS 1D ---
        if method == 'bisection':
            if N != 1: raise ValueError("Bisección solo soporta una variable (N=1).")
            root, iterations = bisection(func_list[0], float(data.get('a')), float(data.get('b')), float(tol), int(max_iter))
            
        elif method == 'secant':
            if N != 1: raise ValueError("Secante solo soporta una variable (N=1).")
            root, iterations = secant_method(func_list[0], data.get('x_prev'), data.get('x_curr'), float(tol), int(max_iter))
            
        # --- MÉTODOS N-VARIABLES ---
        elif method in ['newton_raphson', 'modified_newton', 'fixed_point']:
            if N not in [2, 3]:
                return jsonify({"error": f"{method} requiere N=2 o N=3 variables."}), 400
            
            x0 = data.get('x0')
            if x0 is None or len(x0) != N:
                raise ValueError(f"El método requiere un vector inicial x0 de tamaño {N}.")
            
            if method == 'newton_raphson':
                root, iterations = newton_raphson_system(func_list, x0, N, float(tol), int(max_iter))
                plot_generator = generate_plot_newton
            elif method == 'modified_newton':
                root, iterations = jacobian_frozen_newton(func_list, x0, N, float(tol), int(max_iter))
                plot_generator = None # No gráfico por requisito
            elif method == 'fixed_point':
                root, iterations = fixed_point_system(func_list, x0, N, float(tol), int(max_iter))
                plot_generator = generate_plot_fixed_point

            # Generar gráfica SOLO si N=2 y el método lo permite
            if N == 2 and plot_generator:
                x_min = data.get('x_min', -5); x_max = data.get('x_max', 5)
                y_min = data.get('y_min', -5); y_max = data.get('y_max', 5)
                image_data = plot_generator(func_list, root, float(x_min), float(x_max), float(y_min), float(y_max))
        
        else:
            return jsonify({"error": "Método de búsqueda de raíces no válido."}), 400
            
        # Formato de salida
        if isinstance(root, list):
            formatted_root = {f"x{i+1}": val for i, val in enumerate(root)}
        else:
            formatted_root = float(root)
            
        return jsonify({
            "root": formatted_root, 
            "iterations": iterations,
            "image": image_data
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error en la ejecución del método: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)