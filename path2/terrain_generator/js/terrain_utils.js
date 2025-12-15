/**
 * Terrain utility functions for JavaScript
 * Provides terrain height calculation and other utilities compatible with Python version
 */

class TerrainUtils {
    constructor(worldState, simplex) {
        this.worldState = worldState;
        this.simplex = simplex;
        this.lakes = worldState.lakes || [];
    }

    getCityLimit(angle) {
        const noise = this.simplex.noise2D(Math.cos(angle), Math.sin(angle));
        return 140 + noise * 60;
    }

    getLakeFactor(x, z, lake) {
        const dx = x - lake.x;
        const dz = z - lake.z;
        const cos = Math.cos(-lake.rot);
        const sin = Math.sin(-lake.rot);
        const nx = dx * cos - dz * sin;
        const nz = dx * sin + dz * cos;
        return (nx * nx) / (lake.rx * lake.rx) + (nz * nz) / (lake.rz * lake.rz);
    }

    isInLake(x, z, buffer = 0) {
        for (let lake of this.lakes) {
            const factor = this.getLakeFactor(x, z, lake);
            if (factor < 1.0 + (buffer / lake.rx)) return true;
        }
        return false;
    }

    getTerrainHeight(x, z) {
        const dist = Math.sqrt(x * x + z * z);
        const angle = Math.atan2(z, x);
        const cityLimit = this.getCityLimit(angle);

        let mountainHeight = 0;
        if (dist > cityLimit) {
            let noise = this.simplex.noise2D(x * 0.005, z * 0.005) * 120;
            noise += this.simplex.noise2D(x * 0.015, z * 0.015) * 45;
            noise += this.simplex.noise2D(x * 0.05, z * 0.05) * 10;
            let factor = Math.min(1, (dist - cityLimit) / 40); // TRANSITION_ZONE = 40
            factor = factor * factor * (3 - 2 * factor); // Smoothstep
            mountainHeight = Math.max(0, noise + 10) * factor;
        }

        let lakeBlend = 1.0;
        const waterLevel = -2;
        for (let lake of this.lakes) {
            const factorSq = this.getLakeFactor(x, z, lake);
            const factor = Math.sqrt(factorSq);
            if (factor < 1.0) return waterLevel;
            const bankWidth = 0.4;
            if (factor < 1.0 + bankWidth) {
                let t = (factor - 1.0) / bankWidth;
                t = t * t * (3 - 2 * t); // Smoothstep
                lakeBlend = Math.min(lakeBlend, t);
            }
        }
        return mountainHeight * lakeBlend;
    }

    isOnRoad(x, z) {
        const dist = Math.sqrt(x * x + z * z);
        const angle = Math.atan2(z, x);
        const limit = this.getCityLimit(angle);
        if (dist > limit - 10) return false;
        if (Math.abs(x) < 60 && Math.abs(z) < 60) {
            if (Math.abs(x % 30) < 4 || Math.abs(z % 30) < 4) return true;
        }
        if (Math.abs(z - x) < 6 && dist > 50) return true;
        if (Math.abs(x - Math.sin(z * 0.05) * 20) < 6 && z < -50) return true;
        if (Math.abs(z - Math.sin(x * 0.03) * 30 - 20) < 6 && x > 50) return true;
        const ringNoise = this.simplex.noise2D(x * 0.01, z * 0.01) * 20;
        if (Math.abs(dist - (110 + ringNoise)) < 5) return true;
        return false;
    }
}

