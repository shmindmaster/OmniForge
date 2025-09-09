// Placeholder for STLLoader.js
// In production, replace with actual STLLoader from Three.js examples
console.warn('STLLoader.js placeholder - replace with actual Three.js STLLoader');
if (window.THREE) {
    THREE.STLLoader = function() {
        return {
            load: function(url, onLoad, onProgress, onError) {
                console.log('STLLoader placeholder - would load:', url);
                if (onError) onError(new Error('STLLoader placeholder'));
            }
        };
    };
}