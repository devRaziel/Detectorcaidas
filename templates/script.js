document.getElementById('john').addEventListener('click', function() {
    
    
        const reader = new FileReader();
        reader.onload = function(e) {
            const imageBase64 = e.target.result;
            // Aquí puedes usar imageBase64 para enviar la imagen al servidor
            fetch('http://localhost:5000/video2', {
                

                
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
        
        
    
});