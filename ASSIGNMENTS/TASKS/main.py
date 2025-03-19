from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
tasks = [
    {"id": 1, "title": "Learn Flask", "completed": False},
    {"id": 2, "title": "Build an API", "completed": False},
]

# Get all tasks
@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify(tasks)

# Get a specific task by ID
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task["id"] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({"error": "Task not found"}), 404

# Create a new task
@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.json
    new_task = {
        "id": len(tasks) + 1,
        "title": data.get("title", ""),
        "completed": data.get("completed", False)
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

# Update a task
@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = next((task for task in tasks if task["id"] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    data = request.json
    task["title"] = data.get("title", task["title"])
    task["completed"] = data.get("completed", task["completed"])
    return jsonify(task)

# Delete a task
@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    global tasks
    tasks = [task for task in tasks if task["id"] != task_id]
    return jsonify({"message": "Task deleted successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
