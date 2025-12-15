/**
 * Terrain renderer using Three.js
 * Renders terrain, buildings, trees, and lakes
 */

class TerrainRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id "${containerId}" not found`);
        }

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.worldState = null;
        this.terrainUtils = null;
        this.simplex = null;
        this.pathGroup = null;  // Group to hold all trajectory paths
        this.paths = [];        // Array to store path objects
    }

    initScene() {
        // Scene
        this.scene = new THREE.Scene();
        const bgColor = 0xa0a0a0;
        this.scene.background = new THREE.Color(bgColor);
        // Fog removed as requested
        // this.scene.fog = new THREE.FogExp2(fogColor, 0.002);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            4000
        );
        this.camera.position.set(0, 150, 200);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
        this.scene.add(hemiLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
        dirLight.position.set(150, 300, 100);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 4096;
        dirLight.shadow.mapSize.height = 4096;
        dirLight.shadow.camera.left = -500;
        dirLight.shadow.camera.right = 500;
        dirLight.shadow.camera.top = 500;
        dirLight.shadow.camera.bottom = -500;
        this.scene.add(dirLight);

        // Orbit controls
        if (typeof THREE !== 'undefined' && THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
        } else {
            console.error('THREE.OrbitControls is not available. Make sure OrbitControls.js is loaded before terrain_renderer.js');
            this.controls = null;
        }

        // Initialize path group for trajectories (hidden by default)
        this.pathGroup = new THREE.Group();
        this.pathGroup.visible = false;  // Hidden by default, wait for call to show
        this.scene.add(this.pathGroup);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    }

    initSimplexNoise(seed) {
        // Initialize SimplexNoise with seed
        // Using the same random function as original for compatibility
        let currentSeed = seed;
        function seedRandom(s) {
            currentSeed = s;
        }
        function random() {
            // Mulberry32
            let t = currentSeed += 0x6D2B79F5;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        }
        this.simplex = new SimplexNoise(random);
    }

    renderTerrain() {
        const MAP_SIZE = this.worldState.mapSize || 1000;
        const vertexCount = 500;
        const geometry = new THREE.PlaneGeometry(MAP_SIZE, MAP_SIZE, vertexCount, vertexCount);
        geometry.rotateX(-Math.PI / 2);
        const positions = geometry.attributes.position;
        const colors = [];
        const cGround = new THREE.Color(0x666666);
        const cRoad = new THREE.Color(0x333333);
        const cSand = new THREE.Color(0x8d7e66);

        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const z = positions.getZ(i);
            const h = this.terrainUtils.getTerrainHeight(x, z);
            positions.setY(i, h);

            let isNearLake = false;
            for (let lake of this.worldState.lakes) {
                if (this.terrainUtils.getLakeFactor(x, z, lake) < 1.2) {
                    isNearLake = true;
                    break;
                }
            }

            if (this.terrainUtils.isOnRoad(x, z)) {
                colors.push(cRoad.r, cRoad.g, cRoad.b);
            } else if (isNearLake && h < 3) {
                colors.push(cSand.r, cSand.g, cSand.b);
            } else {
                colors.push(cGround.r, cGround.g, cGround.b);
            }
        }

        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals();

        const ground = new THREE.Mesh(
            geometry,
            new THREE.MeshStandardMaterial({
                vertexColors: true,
                roughness: 0.9,
                flatShading: true
            })
        );
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    renderLakes() {
        this.worldState.lakes.forEach(lake => {
            const waterGeo = new THREE.CircleGeometry(1, 64);
            const wPos = waterGeo.attributes.position;
            for (let i = 1; i < wPos.count; i++) {
                const x = wPos.getX(i);
                const y = wPos.getY(i);
                const angle = Math.atan2(y, x);
                const rNoise = this.simplex.noise2D(Math.cos(angle) * 2, Math.sin(angle) * 2);
                const scale = 1.0 + rNoise * 0.1;
                wPos.setX(i, x * scale);
                wPos.setY(i, y * scale);
            }
            waterGeo.rotateX(-Math.PI / 2);
            waterGeo.scale(lake.rx, 1, lake.rz);
            waterGeo.rotateY(-lake.rot);
            waterGeo.translate(lake.x, 0.2, lake.z);

            const waterMesh = new THREE.Mesh(
                waterGeo,
                new THREE.MeshStandardMaterial({
                    color: 0x2196f3,
                    roughness: 0.1,
                    transparent: true,
                    opacity: 0.8
                })
            );
            this.scene.add(waterMesh);
        });
    }

    renderBuildings() {
        const boxGeo = new THREE.BoxGeometry(1, 1, 1);
        boxGeo.translate(0, 0.5, 0);
        const buildingMat = new THREE.MeshStandardMaterial({ color: 0xdddddd });
        
        const totalBuildings = this.worldState.cityBuildings.length + this.worldState.mountainBuildings.length;
        const buildingsMesh = new THREE.InstancedMesh(boxGeo, buildingMat, totalBuildings);
        buildingsMesh.castShadow = true;
        buildingsMesh.receiveShadow = true;

        const dummy = new THREE.Object3D();
        let bIdx = 0;

        // City buildings
        this.worldState.cityBuildings.forEach(b => {
            dummy.position.set(b.x, 0, b.z);
            dummy.scale.set(b.halfWidth * 2, b.height, b.halfDepth * 2);
            dummy.rotation.y = Math.random() * 0.1; // Visual jitter
            dummy.updateMatrix();
            buildingsMesh.setMatrixAt(bIdx++, dummy.matrix);
        });

        // Mountain buildings
        this.worldState.mountainBuildings.forEach(b => {
            const visualBase = b.baseHeight - 80;
            dummy.position.set(b.x, visualBase, b.z);
            dummy.rotation.set(0, Math.random() * Math.PI, 0);
            dummy.scale.set(b.halfWidth * 2, b.height - visualBase, b.halfDepth * 2);
            dummy.updateMatrix();
            buildingsMesh.setMatrixAt(bIdx++, dummy.matrix);
        });

        this.scene.add(buildingsMesh);
    }

    renderTrees() {
        if (!this.worldState.trees || this.worldState.trees.length === 0) {
            return;
        }

        const treeGeo = new THREE.ConeGeometry(2, 6, 6);
        treeGeo.translate(0, 3, 0);
        const treeMat = new THREE.MeshStandardMaterial({
            color: 0x1b5e20,
            roughness: 0.9,
            flatShading: true
        });

        const treesMesh = new THREE.InstancedMesh(
            treeGeo,
            treeMat,
            this.worldState.trees.length
        );
        treesMesh.castShadow = true;

        const dummy = new THREE.Object3D();
        this.worldState.trees.forEach((tree, i) => {
            // Recalculate terrain height to ensure trees are placed correctly on terrain
            // This fixes the issue where trees might be floating
            const terrainHeight = this.terrainUtils.getTerrainHeight(tree.x, tree.z);
            // Use the terrain height, ensuring trees sit on the ground
            const treeY = terrainHeight;
            
            dummy.position.set(tree.x, treeY, tree.z);
            dummy.scale.set(tree.scale, tree.scale, tree.scale);
            dummy.rotation.set(0, tree.rotation, 0);
            dummy.updateMatrix();
            treesMesh.setMatrixAt(i, dummy.matrix);
        });

        this.scene.add(treesMesh);
    }

    renderUsers() {
        if (!this.worldState.finalUsers || this.worldState.finalUsers.length === 0) {
            return;
        }

        // Create enhanced markers for users
        this.userMarkers = [];

        this.worldState.finalUsers.forEach((user, i) => {
            const userId = i + 1;
            
            // Calculate terrain height at user position to ensure marker is above ground
            const terrainHeight = this.terrainUtils.getTerrainHeight(user.x, user.z);
            // Lift marker above terrain: base at terrain + 5 units, so it's clearly visible
            const markerBaseHeight = terrainHeight + 5;
            // User's actual y position (on building surface) - we'll use the higher of the two
            const userHeight = user.y;
            // Use the maximum to ensure marker is always visible above terrain
            const markerY = Math.max(markerBaseHeight, userHeight + 5);
            
            // Create a group for each user marker
            const userGroup = new THREE.Group();
            
            // Main marker: Larger sphere with bright color
            const sphereGeo = new THREE.SphereGeometry(3, 16, 16);
            const sphereMat = new THREE.MeshBasicMaterial({ 
                color: 0xff4400,
                transparent: true,
                opacity: 0.9
            });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            userGroup.add(sphere);
            
            // Add a ring/cylinder base for better visibility
            const ringGeo = new THREE.CylinderGeometry(4, 4, 0.5, 32);
            const ringMat = new THREE.MeshBasicMaterial({ 
                color: 0xff8800,
                transparent: true,
                opacity: 0.6
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2;
            ring.position.y = -1.5;
            userGroup.add(ring);
            
            // Add a pole/line pointing up for better visibility
            const poleGeo = new THREE.CylinderGeometry(0.3, 0.3, 12, 8);
            const poleMat = new THREE.MeshBasicMaterial({ color: 0xff4400 });
            const pole = new THREE.Mesh(poleGeo, poleMat);
            pole.position.y = 6; // Extended pole
            userGroup.add(pole);
            
            // Position the group - elevated above terrain
            userGroup.position.set(user.x, markerY, user.z);
            this.scene.add(userGroup);
            
            // Create text sprite for label - always visible above terrain
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 256;
            const context = canvas.getContext('2d');
            
            // Draw background with better visibility
            context.fillStyle = 'rgba(0, 0, 0, 0.85)';
            context.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw border
            context.strokeStyle = '#ff4400';
            context.lineWidth = 4;
            context.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
            
            // Draw text
            context.fillStyle = '#ff4400';
            context.font = 'Bold 72px Arial';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(`USER_${userId}`, canvas.width / 2, canvas.height / 2);
            
            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;
            const spriteMaterial = new THREE.SpriteMaterial({ 
                map: texture,
                transparent: true,
                depthTest: false, // Always render on top
                depthWrite: false
            });
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.scale.set(40, 20, 1); // Larger size for better visibility
            sprite.renderOrder = 9999; // Render last, always on top
            
            // Position label high above terrain - ensure it's always visible
            // Use a fixed high position relative to terrain, not marker
            const labelHeight = Math.max(terrainHeight + 30, markerY + 25);
            sprite.position.set(user.x, labelHeight, user.z);
            this.scene.add(sprite);
            
            this.userMarkers.push({
                group: userGroup,
                sprite: sprite,
                userId: userId,
                userX: user.x,
                userZ: user.z,
                userY: user.y
            });
        });
        
        console.log(`Rendered ${this.worldState.finalUsers.length} user markers`);
    }

    async loadAndRender(worldState) {
        this.worldState = worldState;

        // Initialize SimplexNoise with seed
        this.initSimplexNoise(worldState.seed);

        // Initialize terrain utils
        this.terrainUtils = new TerrainUtils(worldState, this.simplex);

        // Render all components
        this.renderTerrain();
        this.renderLakes();
        this.renderBuildings();
        this.renderTrees();
        this.renderUsers();

        console.log('Terrain rendered successfully');
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.controls) {
            this.controls.update();
        }
        
        // Update user label sprites to always face camera and stay above terrain
        if (this.userMarkers) {
            this.userMarkers.forEach(marker => {
                if (marker.sprite) {
                    // Sprite automatically faces camera
                    // Ensure label stays above terrain at all times
                    const terrainHeight = this.terrainUtils.getTerrainHeight(marker.userX, marker.userZ);
                    const minHeight = Math.max(terrainHeight + 30, marker.userY + 25);
                    // Keep label at minimum height
                    if (marker.sprite.position.y < minHeight) {
                        marker.sprite.position.y = minHeight;
                    }
                }
            });
        }
        
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Add a trajectory path to the scene
     * @param {Array} points - Array of points with {x, y, z} coordinates
     * @param {Object} options - Optional configuration
     * @param {number} options.color - Color in hex (default: 0xff0000)
     * @param {number} options.opacity - Opacity 0-1 (default: 0.8)
     * @param {number} options.radius - Tube radius (default: 0.6)
     * @param {number} options.segments - Curve segments (default: 400)
     * @param {string} options.name - Optional name for the path
     * @returns {Object} Path object with show/hide methods
     */
    addPath(points, options = {}) {
        if (!points || points.length < 2) {
            console.warn('Path must have at least 2 points');
            return null;
        }

        const config = {
            color: options.color || 0xff0000,
            opacity: options.opacity !== undefined ? options.opacity : 0.8,
            radius: options.radius || 0.6,
            segments: options.segments || 400,
            name: options.name || `Path_${this.paths.length + 1}`,
            useSpline: options.useSpline !== undefined ? options.useSpline : false, // 默认使用折线以避免穿越障碍
            tension: options.tension !== undefined ? options.tension : 0.0          // 若启用样条，使用低张力减少外扩
        };

        // Convert points to THREE.Vector3
        const threePoints = points.map(p => new THREE.Vector3(p.x, p.y, p.z));

        // Build curve: 默认用折线，避免样条外扩穿过障碍；如需平滑，可传 useSpline=true
        let curve;
        if (config.useSpline) {
            curve = new THREE.CatmullRomCurve3(threePoints, false, 'catmullrom', config.tension);
        } else {
            const curvePath = new THREE.CurvePath();
            for (let i = 0; i < threePoints.length - 1; i++) {
                curvePath.add(new THREE.LineCurve3(threePoints[i], threePoints[i + 1]));
            }
            curve = curvePath;
        }

        // Create tube geometry
        const geometry = new THREE.TubeGeometry(curve, config.segments, config.radius, 6, false);
        const material = new THREE.MeshLambertMaterial({
            color: config.color,
            transparent: true,
            opacity: config.opacity
        });

        const mesh = new THREE.Mesh(geometry, material);
        this.pathGroup.add(mesh);

        // Create path object
        const pathObj = {
            name: config.name,
            mesh: mesh,
            curve: curve,
            visible: false,
            show: () => {
                mesh.visible = true;
                pathObj.visible = true;
            },
            hide: () => {
                mesh.visible = false;
                pathObj.visible = false;
            },
            remove: () => {
                this.pathGroup.remove(mesh);
                geometry.dispose();
                material.dispose();
                const index = this.paths.indexOf(pathObj);
                if (index > -1) {
                    this.paths.splice(index, 1);
                }
            }
        };

        // Initially hidden
        mesh.visible = false;

        this.paths.push(pathObj);
        return pathObj;
    }

    /**
     * Load paths from JSON data
     * @param {Object|Array} pathData - Path data (can be array of paths or object with paths array)
     * @param {Object} defaultOptions - Default options for paths
     */
    loadPaths(pathData, defaultOptions = {}) {
        let pathsArray = [];

        // Handle different data formats
        if (Array.isArray(pathData)) {
            pathsArray = pathData;
        } else if (pathData.paths && Array.isArray(pathData.paths)) {
            pathsArray = pathData.paths;
        } else if (pathData.points && Array.isArray(pathData.points)) {
            // Single path with points
            pathsArray = [{ points: pathData.points, ...pathData }];
        } else {
            console.warn('Invalid path data format');
            return [];
        }

        const loadedPaths = [];
        pathsArray.forEach((path, index) => {
            const points = path.points || path;
            if (!Array.isArray(points) || points.length < 2) {
                console.warn(`Invalid path at index ${index}, skipping`);
                return;
            }

            const options = {
                ...defaultOptions,
                ...path,
                name: path.name || `Path_${index + 1}`
            };

            const pathObj = this.addPath(points, options);
            if (pathObj) {
                loadedPaths.push(pathObj);
            }
        });

        return loadedPaths;
    }

    /**
     * Show all paths
     */
    showPaths() {
        this.pathGroup.visible = true;
        this.paths.forEach(path => path.show());
    }

    /**
     * Hide all paths
     */
    hidePaths() {
        this.pathGroup.visible = false;
        this.paths.forEach(path => path.hide());
    }

    /**
     * Clear all paths
     */
    clearPaths() {
        this.paths.forEach(path => path.remove());
        this.paths = [];
    }

    /**
     * Show specific path by name or index
     * @param {string|number} identifier - Path name or index
     */
    showPath(identifier) {
        const path = this.getPath(identifier);
        if (path) {
            path.show();
            this.pathGroup.visible = true;
        }
    }

    /**
     * Hide specific path by name or index
     * @param {string|number} identifier - Path name or index
     */
    hidePath(identifier) {
        const path = this.getPath(identifier);
        if (path) {
            path.hide();
        }
    }

    /**
     * Get path by name or index
     * @param {string|number} identifier - Path name or index
     * @returns {Object|null} Path object or null
     */
    getPath(identifier) {
        if (typeof identifier === 'number') {
            return this.paths[identifier] || null;
        } else if (typeof identifier === 'string') {
            return this.paths.find(p => p.name === identifier) || null;
        }
        return null;
    }

    start() {
        this.animate();
    }
}

