const operationContainer = document.getElementById('operation-container');
const resultOutput = document.getElementById('result-output');
const plotImage = document.getElementById('plot-image'); // Nuevo elemento para la imagen
const API_URL = 'http://127.0.0.1:5000'; // URL base de Flask
let currentMethod = '';    // Para sistemas (Gauss, Jacobi, etc.)

// ==========================================================
// LÓGICA PARA OPERACIONES BÁSICAS (Suma, Mult, Det, Inv)
// ==========================================================

function showMatrixSetup(operation) {
    currentOperation = operation;
    currentMethod = ''; // Limpiar el método de sistema
    resultOutput.textContent = '';
    let html = `<h2>${operation.toUpperCase()} de Matrices</h2>`;

    if (operation === 'inverse' || operation === 'determinant') {
        html += `
            <label for="rowsA">Tamaño (N x N):</label>
            <input type="number" id="rowsA" value="3" min="1" oninput="generateMatrixInputs('A', this.value, this.value)">
            <div id="matrix-inputs-A"></div>
            <button onclick="generateMatrixInputs('A', document.getElementById('rowsA').value, document.getElementById('rowsA').value)">Generar A</button>
        `;
    } 
    else if (operation === 'sum' || operation === 'multiply') {
        html += `
            <h3>Matriz A</h3>
            <label for="rowsA">Filas A:</label>
            <input type="number" id="rowsA" value="2" min="1">
            <label for="colsA">Columnas A:</label>
            <input type="number" id="colsA" value="2" min="1">
            <button onclick="generateMatrixInputs('A', document.getElementById('rowsA').value, document.getElementById('colsA').value)">Generar A</button>
            <div id="matrix-inputs-A"></div>

            <h3>Matriz B</h3>
            <label for="rowsB">Filas B:</label>
            <input type="number" id="rowsB" value="2" min="1">
            <label for="colsB">Columnas B:</label>
            <input type="number" id="colsB" value="2" min="1">
            <button onclick="generateMatrixInputs('B', document.getElementById('rowsB').value, document.getElementById('colsB').value)">Generar B</button>
            <div id="matrix-inputs-B"></div>
        `;
    }

    html += '<button onclick="calculateMatrix()">Calcular Operación</button>';
    operationContainer.innerHTML = html;
    // Generación inicial para la primera matriz (solo si es Determinante/Inversa)
    if (operation === 'inverse' || operation === 'determinant') {
        generateMatrixInputs('A', 3, 3);
    }
}

function generateMatrixInputs(matrixName, rows, cols) {
    const container = document.getElementById(`matrix-inputs-${matrixName}`);
    rows = parseInt(rows);
    cols = parseInt(cols);
    let html = `<table>`;

    for (let i = 0; i < rows; i++) {
        html += `<tr>`;
        for (let j = 0; j < cols; j++) {
            // Valor por defecto: 1 en la diagonal, 0 en el resto
            const defaultValue = (i === j) ? 1 : 0;
            html += `<td><input type="number" id="${matrixName}_${i}_${j}" value="${defaultValue}" step="any" required></td>`;
        }
        html += `</tr>`;
    }

    html += `</table>`;
    container.innerHTML = html;
}

function getMatrixData(matrixName) {
    const rowsInput = document.getElementById(`rows${matrixName}`);
    let colsInput;
    
    if (matrixName === 'A' && (currentOperation === 'inverse' || currentOperation === 'determinant')) {
        colsInput = rowsInput; // Para matrices cuadradas (N x N)
    } else {
        colsInput = document.getElementById(`cols${matrixName}`);
    }
    
    if (!rowsInput || !colsInput) return null;

    const rows = parseInt(rowsInput.value);
    const cols = parseInt(colsInput.value);
    const matrix = [];

    for (let i = 0; i < rows; i++) {
        const row = [];
        for (let j = 0; j < cols; j++) {
            const input = document.getElementById(`${matrixName}_${i}_${j}`);
            if (!input) {
                throw new Error(`Matriz ${matrixName} incompleta. Asegúrate de generar la matriz primero.`);
            }
            row.push(input.value);
        }
        matrix.push(row);
    }
    return matrix;
}

async function calculateMatrix() {
    try {
        const matrixA = getMatrixData('A');
        let matrixB = null;
        if (currentOperation === 'sum' || currentOperation === 'multiply') {
            matrixB = getMatrixData('B');
        }

        if (!matrixA) {
            throw new Error('Matriz A no generada o incompleta.');
        }

        const payload = {
            operation: currentOperation,
            matrixA: matrixA,
            matrixB: matrixB
        };

        const response = await fetch(`${API_URL}/api/matrix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            if (Array.isArray(result.result)) {
                // Formato de matriz (Suma, Multiplicación, Inversa)
                const formattedMatrix = result.result.map(row => 
                    row.map(val => val.toFixed(6).padStart(10)).join(' ')
                ).join('\n');
                resultOutput.textContent = formattedMatrix;
            } else {
                // Formato de escalar (Determinante)
                resultOutput.textContent = result.result.toFixed(8);
            }
        } else {
            resultOutput.textContent = `ERROR: ${result.error}`;
        }
    } catch (error) {
        resultOutput.textContent = `ERROR: ${error.message}`;
        console.error("Error en la solicitud de matriz:", error);
    }
}

// ==========================================================
// LÓGICA PARA SISTEMAS DE ECUACIONES (Gauss, Jacobi, etc.)
// ==========================================================

function showSystemSetup(method) {
    currentMethod = method;
    currentOperation = ''; // Limpiar la operación de matriz
    resultOutput.textContent = '';
    
    let title = '';
    let isIterative = false;

    if (method === 'gauss_elimination') title = 'Triangulación de Gauss';
    else if (method === 'gauss_jordan') title = 'Gauss-Jordan';
    else if (method === 'jacobi') { title = 'Método de Jacobi'; isIterative = true; }
    else if (method === 'gauss_seidel') { title = 'Método de Gauss-Seidel'; isIterative = true; }
    
    let html = `<h2>${title} ($A\mathbf{x} = \mathbf{b}$)</h2>`;

    html += `
        <label for="sizeN">Tamaño del Sistema (N x N):</label>
        <input type="number" id="sizeN" value="3" min="2" oninput="generateSystemInputs(this.value)">
        <p>Introduce la **Matriz de Coeficientes A** y el **Vector b**.</p>
    `;
    
    if (isIterative) {
        html += `
            <h3>Parámetros Iterativos</h3>
            <label for="maxIter">Máx. Iteraciones:</label>
            <input type="number" id="maxIter" value="100" min="10">
            <label for="tolerance">Tolerancia ($\epsilon$):</label>
            <input type="text" id="tolerance" value="0.000001">
            <p class="warning">Recomendación: Para estos métodos, intente hacer la matriz **Diagonal Dominante** para asegurar la convergencia.</p>
        `;
    }

    html += `
        <div id="system-inputs-container"></div>
        <button onclick="calculateSystem()">Calcular Solución</button>
    `;

    operationContainer.innerHTML = html;
    generateSystemInputs(document.getElementById('sizeN').value);
}

function generateSystemInputs(N) {
    N = parseInt(N);
    const container = document.getElementById('system-inputs-container');
    let html = `
        <div class="system-io">
            <div>
                <h3>Matriz A (Coeficientes)</h3>
                <table>
    `;

    // Generación de la Matriz A (ejemplo: diagonal dominante para pruebas iterativas)
    for (let i = 0; i < N; i++) {
        html += `<tr>`;
        for (let j = 0; j < N; j++) {
            const defaultValue = (i === j) ? 10 + i : 1; // Un ejemplo de matriz que podría ser diagonal dominante
            html += `<td><input type="number" id="A_${i}_${j}" value="${defaultValue}" step="any" required></td>`;
        }
        html += `</tr>`;
    }
    html += `
                </table>
            </div>
            <div>
                <h3>Vector b</h3>
                <table>
    `;

    // Generación del Vector b
    for (let i = 0; i < N; i++) {
        html += `<tr><td><input type="number" id="b_${i}" value="${N + i}" step="any" required></td></tr>`;
    }
    
    html += `
                </table>
            </div>
        </div>
    `;
    container.innerHTML = html;
}

async function calculateSystem() {
    try {
        const N = parseInt(document.getElementById('sizeN').value);
        const matrixA = [];
        const vectorB = [];
        
        // 1. Recolectar Matriz A
        for (let i = 0; i < N; i++) {
            const row = [];
            for (let j = 0; j < N; j++) {
                const input = document.getElementById(`A_${i}_${j}`);
                if (!input) throw new Error("Matriz A incompleta o no generada.");
                row.push(input.value);
            }
            matrixA.push(row);
        }

        // 2. Recolectar Vector b
        for (let i = 0; i < N; i++) {
            const input = document.getElementById(`b_${i}`);
            if (!input) throw new Error("Vector b incompleto.");
            vectorB.push(input.value);
        }

        const payload = {
            method: currentMethod,
            matrixA: matrixA,
            vectorB: vectorB
        };

        // 3. Agregar parámetros iterativos si aplica
        if (currentMethod === 'jacobi' || currentMethod === 'gauss_seidel') {
            payload.maxIter = parseInt(document.getElementById('maxIter').value);
            payload.tolerance = parseFloat(document.getElementById('tolerance').value);
        }

        const response = await fetch(`${API_URL}/api/system`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            const formattedResult = Object.entries(result.result)
                .map(([key, value]) => `${key} = ${value.toFixed(10)}`)
                .join('\n');
            
            if (result.iterations !== undefined && result.iterations !== null) {
                resultOutput.textContent = `Solución encontrada en ${result.iterations} iteraciones.\n\n${formattedResult}`;
            } else {
                resultOutput.textContent = formattedResult;
            }

        } else {
            resultOutput.textContent = `ERROR DEL SERVIDOR: ${result.error}`;
        }
    } catch (error) {
        resultOutput.textContent = `ERROR EN EL CLIENTE: ${error.message}`;
        console.error("Error en la solicitud del sistema:", error);
    }
}



// ==========================================================
// LÓGICA PARA BÚSQUEDA DE RAÍCES (Bisección y Newton-Raphson)
// ==========================================================





function generateNewtonInputs(N) {
    N = parseInt(N);
    const funcContainer = document.getElementById('function-inputs-container');
    const x0Container = document.getElementById('x0-inputs-container');
    
    if (N === 1) {
        // Modo Bisección/Newton de 1 variable
        funcContainer.innerHTML = `<input type="text" id="func_0" value="x**3 - 2*x - 5" style="width: 250px;" required>`;
        x0Container.innerHTML = '';
        return;
    }

    // Modo Newton-Raphson para N variables
    let funcHtml = '';
    let x0Html = '';
    
    for (let i = 0; i < N; i++) {
        const varName = `x${i + 1}`;
        funcHtml += `
            <div style="margin-bottom: 5px;">
                <label for="func_${i}">F${i + 1} = 0:</label>
                <input type="text" id="func_${i}" value="${N === 2 ? (i === 0 ? 'x1**2 + x2**2 - 4' : 'x1*x2 - 1') : `x${i+1}`}" style="width: 250px;" required>
            </div>
        `;
        x0Html += `
            <div style="margin-bottom: 5px;">
                <label for="x0_${i}">x${i + 1}:</label>
                <input type="number" id="x0_${i}" value="${i + 1}" step="any" required>
            </div>
        `;
    }
    
    funcContainer.innerHTML = funcHtml;
    x0Container.innerHTML = x0Html;
}

// Auxiliar: Genera los campos de funciones y x0 para métodos de sistemas (N >= 2)
function generateSystemRootInputs(method, N) {
    N = parseInt(N);
    const funcContainer = document.getElementById('function-inputs-container');
    const x0Container = document.getElementById('x0-inputs-container');
    
    // Si N=1, generamos un único campo para f(x) y no hay campos x0
    if (N === 1) {
        funcContainer.innerHTML = `
            <input type="text" id="func_0" value="x**3 - 2*x - 5" style="width: 250px;" required>
            <input type="hidden" id="method_type" value="1d">
        `;
        x0Container.innerHTML = '';
        return;
    }

    // Modo Sistema (N=2, 3)
    let funcHtml = '';
    let x0Html = '';
    
    for (let i = 0; i < N; i++) {
        // Valores de ejemplo preestablecidos
        let funcValue = `x${i + 1}`;
        let x0Value = (i === 0 || i === 1) ? 0.5 : 1; // 0.5 para x1, x2, 1 para x3
        
        if (method === 'newton_raphson' || method === 'modified_newton') {
            // Ejemplo F(x) = 0
            funcValue = N === 2 ? (i === 0 ? 'x1**2 + x2**2 - 4' : 'x1*x2 - 1') : `x${i+1}`;
        } else if (method === 'fixed_point') {
            // Ejemplo G(x) = x. CORREGIDO: Usamos cos/sin directamente (asumiendo están en el backend)
            funcValue = N === 2 ? (i === 0 ? 'cos(x2)' : 'sin(x1)') : `x${i+1}`; 
            x0Value = 0.5; // Según el ejemplo de la imagen
        }

        funcHtml += `
            <div style="margin-bottom: 5px;">
                <label for="func_${i}">F${i + 1} = 0 / G${i+1}:</label>
                <input type="text" id="func_${i}" value="${funcValue}" style="width: 250px;" required>
            </div>
        `;
        x0Html += `
            <div style="margin-bottom: 5px;">
                <label for="x0_${i}">x${i + 1}:</label>
                <input type="number" id="x0_${i}" value="${x0Value}" step="any" required>
            </div>
        `;
    }
    
    funcContainer.innerHTML = funcHtml + `<input type="hidden" id="method_type" value="system">`;
    x0Container.innerHTML = x0Html;
}



function showRootSetup(method) {
    currentMethod = method;
    resultOutput.textContent = '';
    plotImage.style.display = 'none';
    plotImage.src = '';
    
    let is1D = method === 'bisection' || method === 'secant';
    let isSystem = method === 'fixed_point' || method === 'newton_raphson' || method === 'modified_newton';
    let title = '';

    if (method === 'bisection') title = 'Bisección (1D)';
    else if (method === 'secant') title = 'Secante (1D)';
    else if (method === 'fixed_point') title = 'Punto Fijo (Sistema G(x)=x)';
    else if (method === 'newton_raphson') title = 'Newton-Raphson (Sistema F(x)=0)';
    else if (method === 'modified_newton') title = 'Newton Modificado (Jacobiano Congelado)';
    
    let html = `<h2>Método de ${title}</h2>`;

    // Selector de variables para sistemas (N=2, N=3)
    if (isSystem) {
        let defaultN = 2;
        html += `
            <div class="root-params">
                <label for="num_vars">Variables (N):</label>
                <select id="num_vars" onchange="generateSystemRootInputs('${method}', this.value)">
                    <option value="2" ${defaultN === 2 ? 'selected' : ''}>2 Variables (x1, x2) - Gráfico disponible</option>
                    <option value="3" ${defaultN === 3 ? 'selected' : ''}>3 Variables (x1, x2, x3)</option>
                </select>
            </div>
        `;
    } else {
        html += `<input type="hidden" id="num_vars" value="1">`; // N=1 para 1D
    }

    // Inicializa la función para Bisección/Secante aquí, antes de los demás campos
    if (is1D) {
        generateSystemRootInputs(method, 1);
    }
    
    html += `
        <div class="root-params">
            <h3>${isSystem ? (method === 'fixed_point' ? 'Funciones G(x) = x' : 'Funciones F(x) = 0') : 'Función f(x)'}</h3>
            <div id="function-inputs-container"></div>
            
            <h3>Parámetros de Convergencia</h3>
            <label for="maxIter">Máx. Iteraciones:</label>
            <input type="number" id="maxIter" value="50" min="10">
            <label for="tolerance">Tolerancia ($\epsilon$):</label>
            <input type="text" id="tolerance" value="0.000001">
        </div>
    `;

    // Parámetros de Punto Inicial
    if (method === 'bisection') {
        html += `
            <h3>Intervalo [a, b]</h3>
            <label for="a">Límite 'a':</label>
            <input type="number" id="a" value="2" step="any">
            <label for="b">Límite 'b':</label>
            <input type="number" id="b" value="3" step="any">
            <p class="warning">Bisección: El intervalo debe cumplir $f(a) \cdot f(b) < 0$.</p>
        `;
    } else if (method === 'secant') {
        html += `
            <h3>Puntos Iniciales</h3>
            <label for="x_prev">x_{i-1}:</label>
            <input type="number" id="x_prev" value="2" step="any">
            <label for="x_curr">x_i:</label>
            <input type="number" id="x_curr" value="3" step="any">
        `;
    } else if (isSystem) {
        html += `
            <h3>Punto Inicial $\mathbf{x}_0$</h3>
            <div id="x0-inputs-container" style="display: flex; gap: 15px;"></div>
        `;
        
        // Parámetros de la gráfica (Solo N=2 y no es Modified Newton)
        if (method !== 'modified_newton') {
            html += `
                <div id="plot-params">
                    <h3>Visualización (Solo N=2)</h3>
                    <label for="x_min">X1 Mín:</label>
                    <input type="number" id="x_min" value="-1" step="any">
                    <label for="x_max">X1 Máx:</label>
                    <input type="number" id="x_max" value="2" step="any">
                    <label for="y_min">X2 Mín:</label>
                    <input type="number" id="y_min" value="-1" step="any">
                    <label for="y_max">X2 Máx:</label>
                    <input type="number" id="y_max" value="2" step="any">
                </div>
            `;
        }
    }
    
    html += '<button onclick="calculateRoot()">Calcular Raíz/Punto Fijo</button>';
    operationContainer.innerHTML = html;

    // Inicializa campos para sistemas después de cargar el HTML
    if (isSystem) {
        generateSystemRootInputs(method, 2); // Carga por defecto N=2
    }
}

async function calculateRoot() {
    try {
        const N_vars = parseInt(document.getElementById('num_vars').value);
        const method_type_input = document.getElementById('method_type');
        const method_type = method_type_input ? method_type_input.value : 'system'; // 'system' si el input no existe (N>=2)
        
        const funcList = [];
        
        // Recolectar funciones
        for (let i = 0; i < N_vars; i++) {
            const funcInput = document.getElementById(`func_${i}`);
            if (!funcInput || funcInput.value.trim() === '') {
                 throw new Error(`La función ${N_vars > 1 ? 'F/G' + (i+1) : 'f(x)'} no puede estar vacía.`);
            }
            funcList.push(funcInput.value);
        }

        const payload = {
            method: currentMethod,
            functions: funcList,
            N: N_vars,
            tolerance: document.getElementById('tolerance').value,
            maxIter: parseInt(document.getElementById('maxIter').value),
        };

        if (method_type === '1d') {
            if (currentMethod === 'bisection') {
                payload.a = document.getElementById('a').value;
                payload.b = document.getElementById('b').value;
            } else if (currentMethod === 'secant') {
                payload.x_prev = document.getElementById('x_prev').value;
                payload.x_curr = document.getElementById('x_curr').value;
            }
        } else if (method_type === 'system') {
            const x0 = [];
            for (let i = 0; i < N_vars; i++) {
                const x0Input = document.getElementById(`x0_${i}`);
                if (!x0Input) throw new Error(`Punto inicial x${i+1} incompleto.`);
                x0.push(parseFloat(x0Input.value));
            }
            payload.x0 = x0;
            
            // Parámetros de la gráfica (Solo si N=2 y no es Modified Newton)
            if (N_vars === 2 && currentMethod !== 'modified_newton') {
                payload.x_min = document.getElementById('x_min').value;
                payload.x_max = document.getElementById('x_max').value;
                payload.y_min = document.getElementById('y_min').value;
                payload.y_max = document.getElementById('y_max').value;
            }
        }
        
        const response = await fetch(`${API_URL}/api/roots`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            let outputText = `Iteraciones = ${result.iterations}\n\n`;
            
            if (typeof result.root === 'object') {
                // Para N > 1 (vector)
                outputText += "Vector Solución (x):\n";
                outputText += Object.entries(result.root)
                    .map(([key, value]) => `${key} = ${value.toFixed(10)}`)
                    .join('\n');
            } else {
                // Para N = 1 (escalar)
                outputText += `Raíz (x) = ${result.root.toFixed(10)}`;
            }
            
            resultOutput.textContent = outputText;

            // Manejo de la imagen (condicional a N=2 y NO ser Newton Modificado)
            if (N_vars === 2 && currentMethod !== 'modified_newton' && result.image) {
                plotImage.src = `data:image/png;base64,${result.image}`;
                plotImage.style.display = 'block';
            } else {
                plotImage.style.display = 'none';
                plotImage.src = '';
            }

        } else {
            resultOutput.textContent = `ERROR DEL SERVIDOR: ${result.error}`;
            plotImage.style.display = 'none';
            plotImage.src = '';
        }
    } catch (error) {
        resultOutput.textContent = `ERROR EN EL CLIENTE: ${error.message}`;
        console.error("Error en la solicitud de raíces:", error);
    }
}