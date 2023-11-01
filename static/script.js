window.addEventListener('DOMContentLoaded', function() {
    const salaryInput = document.getElementById('est_salary');
    salaryInput.value = '0';
    document.querySelector('form').addEventListener('submit', function(event) {
        if (salaryInput.value.trim() === '') {
            salaryInput.value = '0';
        }
    });
});
