/**
 * Terrain data loader
 * Loads terrain data from JSON file
 */

class TerrainLoader {
    constructor(dataPath = 'data/terrain_data.json') {
        this.dataPath = dataPath;
        this.worldState = null;
        this.loaded = false;
    }

    async load() {
        try {
            const response = await fetch(this.dataPath);
            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error(`File not found: ${this.dataPath}. Make sure you have generated terrain data first.`);
                }
                throw new Error(`Failed to load ${this.dataPath}: ${response.status} ${response.statusText}`);
            }
            const data = await response.json();
            
            // Validate essential fields
            // Note: seed could be 0 (falsy) in theory, so check undefined/null instead of truthiness.
            const hasSeed = data.seed !== undefined && data.seed !== null;
            const hasCityBuildings = Array.isArray(data.cityBuildings);
            const hasLakes = Array.isArray(data.lakes);
            if (!hasSeed || !hasCityBuildings || !hasLakes) {
                throw new Error('Invalid terrain data format: missing required fields (seed/lakes/cityBuildings)');
            }
            
            this.worldState = data;
            this.loaded = true;
            console.log('Terrain data loaded successfully');
            return this.worldState;
        } catch (error) {
            console.error('Error loading terrain data:', error);
            // Check if it's a CORS or network error
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error.name === 'TypeError') {
                throw new Error('CORS/Network error. Please use a local web server. Run: python3 -m http.server 8001');
            }
            throw error;
        }
    }

    getWorldState() {
        if (!this.loaded) {
            throw new Error('Terrain data not loaded. Call load() first.');
        }
        return this.worldState;
    }

    isLoaded() {
        return this.loaded;
    }
}

