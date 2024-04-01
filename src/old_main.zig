const std = @import("std");

const ray = @cImport({
    @cInclude("raylib.h");
    @cInclude("rlgl.h");
    @cInclude("raymath.h");
});

const interior = @import("interior.zig");
const mclone = @import("mclone.zig");

extern fn update_camera_first_person(camera: *ray.Camera) callconv(.C) void;
extern fn init_camera_first_person(camera: *ray.Camera) callconv(.C) void;

const RayHit = struct {
    distance: f32,
    // hit position is just ray start + ray direction * distance
    // oh a hitnormal might be useful
};

pub fn GetCollisionRayTriangle(raycast: ray.Ray, a: ray.Vector3, b: ray.Vector3, c: ray.Vector3) ?RayHit {
    // https://github.com/raysan5/raylib/pull/212/files
    const EPSILON = 0.000001;

    const e1 = ray.Vector3Subtract(b, a);
    const e2 = ray.Vector3Subtract(c, a);

    const p = ray.Vector3CrossProduct(raycast.direction, e2);

    const det = ray.Vector3DotProduct(e1, p);

    if(det > -EPSILON and det < EPSILON) return null;
    const inv_det = 1 / det;

    const tv = ray.Vector3Subtract(raycast.position, a);

    const u = ray.Vector3DotProduct(tv, p) * inv_det;

    if(u < 0 or u > 1) return null;

    const q = ray.Vector3CrossProduct(tv, e1);

    const v = ray.Vector3DotProduct(raycast.direction, q) * inv_det;

    if(v < 0 or (u + v) > 1) return null;

    const t = ray.Vector3DotProduct(e2, q) * inv_det;

    if(t > EPSILON) {
        // TODO if normal != triangle normal, return null
        return RayHit{
            .distance = t,
        };
    }

    return null;
}

pub fn toVec3(position: std.meta.Vector(3, f32)) ray.Vector3 {
    return .{.x = position[0], .y = position[1], .z = position[2]};
}

pub fn colorForInt(int: usize) ray.Color {
    var r = std.rand.Xoshiro256.init(int);

    return .{
        .r = @as(u8, r.random().int(u6)) + 0b11000000,
        .g = @as(u8, r.random().int(u6)) + 0b11000000,
        .b = @as(u8, r.random().int(u6)) + 0b11000000,
        .a = 255,
    };
}

fn vec2f32toVec2cint(vec: interior.Vec2f32) std.meta.Vector(2, c_int) {
    return .{std.math.lossyCast(c_int, vec[0]), std.math.lossyCast(c_int, vec[1])};
}

fn toMinimap(vec: interior.Vec2f32) interior.Vec2f32 {
    const room_scale = 20;
    return vec * interior.Vec2f32{room_scale, room_scale} + interior.Vec2f32{100, 100};
}

fn wall3d(ul_: interior.Vec2f32, br_: interior.Vec2f32, xy: u1, comptime room_scale: anytype) void {
    const m = struct{fn m(a: f32, c: f32, b: u1) interior.Vec2f32 {
        return switch(b) {
            0 => .{c, -a},
            1 => .{a, c},
        };
    }}.m;

    const ul = @minimum(ul_, br_);
    const br = @maximum(ul_, br_);
    const ula = ul + m(-0.05, -0.05, xy);
    const ulb = ul + m( 0.05, -0.05, xy);
    const bra = br + m(-0.05, 0.05, xy);
    const brb = br + m( 0.05, 0.05, xy);

    wall3dR(ulb, ula, room_scale);
    wall3dR(brb, ulb, room_scale);
    wall3dR(bra, brb, room_scale);
    wall3dR(ula, bra, room_scale);
}

fn interpolateInt(a: anytype, b: @TypeOf(a), progress: f64) @TypeOf(a) {
    const OO = @TypeOf(a);

    const floa = @intToFloat(f64, a);
    const flob = @intToFloat(f64, b);

    return @floatToInt(OO, floa + (flob - floa) * progress);
}

fn mixColor(a: ray.Color, b: ray.Color, progress: f64) ray.Color {
    return .{
        .r = interpolateInt(a.r, b.r, progress),
        .g = interpolateInt(a.g, b.g, progress),
        .b = interpolateInt(a.b, b.b, progress),
        .a = interpolateInt(a.a, b.a, progress),
    };
}

fn DrawQuad3D(v0: ray.Vector3, v1: ray.Vector3, v2: ray.Vector3, v3: ray.Vector3, c0: ray.Color) void {
    DrawTriangle3D(v0, v1, v2, c0);
    DrawTriangle3D(v3, v2, v1, c0);
}

var drawn_vertex_count: usize = 0;
var drawn_batches_count: usize = 0;
var max_vertex_count: usize = 0b111111111111010; // not sure what the significance of this
// number is. it's close to a power of 2. this is probably system-specific or something.
// raylib should be handling this for me but for some reason checkRenderBatchLimit
// and rlVertex3f are not working properly

fn resetTris() void {
    ray.rlDrawRenderBatchActive();
    drawn_vertex_count = 0;
    drawn_batches_count += 1;
}

fn addVertices(count: usize) void {
    if(ray.rlCheckRenderBatchLimit(@intCast(c_int, count))) {
        drawn_vertex_count = 0;
        drawn_batches_count += 1;
        @panic("wow! checkrenderbatchlimit works. you can probably just delete this line.");
    }else{
        if(drawn_vertex_count > max_vertex_count) resetTris();
        drawn_vertex_count += count;
    }
}

fn DrawTriangle3D(v1: ray.Vector3, v2: ray.Vector3, v3: ray.Vector3, color: ray.Color) void {
    const normal = ray.Vector3Normalize(ray.Vector3CrossProduct(
        ray.Vector3Subtract(v2, v1),
        ray.Vector3Subtract(v3, v1),
    ));
    // const dot = ray.Vector3DotProduct(normal, .{.x = -0.5, .y = -0.2, .z = 1});
    const dot = ray.Vector3DotProduct(normal, .{.x = -0.2, .y = 1, .z = 0.5});

    // life hack:
    // can't get the shader to give you the normal value?
    // try giving up!
    const color_adjust = std.meta.Vector(3, f32){
        dot * 0.5 + 0.5,
        dot * 0.5 + 0.5,
        dot * 0.5 + 0.5,
    } * std.meta.Vector(3, f32){255, 255, 255};
    const res_color: ray.Color = mixColor(color, .{
        .r = std.math.lossyCast(u8, color_adjust[0]),
        .g = std.math.lossyCast(u8, color_adjust[1]),
        .b = std.math.lossyCast(u8, color_adjust[2]),
        .a = color.a,
    }, 0.2);

    if(linesmode) {
        addVertices(8 * 3); // 8 because of some alignment thing idk
        ray.rlBegin(ray.RL_LINES);
        defer ray.rlEnd();

        ray.rlColor4ub(res_color.r, res_color.g, res_color.b, res_color.a);

        ray.rlVertex3f(v1.x, v1.y, v1.z);
        ray.rlVertex3f(v2.x, v2.y, v2.z);

        ray.rlVertex3f(v2.x, v2.y, v2.z);
        ray.rlVertex3f(v3.x, v3.y, v3.z);
        
        ray.rlVertex3f(v3.x, v3.y, v3.z);
        ray.rlVertex3f(v1.x, v1.y, v1.z);

        return;
    }

    addVertices(3);
    
    ray.rlBegin(ray.RL_TRIANGLES);
    defer ray.rlEnd();
    
    ray.rlColor4ub(res_color.r, res_color.g, res_color.b, res_color.a);
    
    // spoiler alert: this function is actually ignored by raylib on opengl 3.3. the
    // only reason lighting demos work as far as I can tell is because they use
    // DrawModel/DrawMesh which feed normals to the shader correctly.
    // ray.rlNormal3f(normal.x, normal.y, normal.z);

    ray.rlVertex3f(v1.x, v1.y, v1.z);
    ray.rlVertex3f(v2.x, v2.y, v2.z);
    ray.rlVertex3f(v3.x, v3.y, v3.z);
}

fn wall3dR(ul: interior.Vec2f32, br: interior.Vec2f32, comptime room_scale: anytype) void {
    const c0 = ray.GRAY;

    const scaled_ul = ul * interior.Vec2f32{room_scale, room_scale};
    const scaled_br = br * interior.Vec2f32{room_scale, room_scale};
    DrawQuad3D(
        .{.x = scaled_ul[0], .y = 0, .z = scaled_ul[1]},
        .{.x = scaled_br[0], .y = 0, .z = scaled_br[1]},
        .{.x = scaled_ul[0], .y = 6, .z = scaled_ul[1]},
        .{.x = scaled_br[0], .y = 6, .z = scaled_br[1]},
        c0,
    );
}

fn wall2d(ul: interior.Vec2f32, br: interior.Vec2f32) void {
    ray.DrawLine(
        vec2f32toVec2cint(toMinimap(ul))[0],
        vec2f32toVec2cint(toMinimap(ul))[1],
        vec2f32toVec2cint(toMinimap(br))[0],
        vec2f32toVec2cint(toMinimap(br))[1],
        ray.BLACK,
    );
}

var linesmode = false;

fn colorToVec3(color: ray.Color) ray.Vector3 {
    return .{
        .x = @intToFloat(f32, color.r) / 255,
        .y = @intToFloat(f32, color.g) / 255,
        .z = @intToFloat(f32, color.b) / 255,
    };
}

pub fn main() !void {
    const window_width = 1920;
    const window_height = 1080;
    std.log.info("started.", .{});

    var r = std.rand.Xoshiro256.init(1);
    std.log.info("int: {d}", .{r.random().int(u8)});

    ray.InitWindow(window_width, window_height, "demo");
    defer ray.CloseWindow();

    var camera = ray.Camera{
        .position = .{ .x = 4, .y = 2, .z = 4 },
        .target = .{ .x = 0, .y = 1.8, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 90,
        .projection = ray.CAMERA_PERSPECTIVE,
    };

    const shader = ray.LoadShaderFromMemory(
        \\#version 330
        \\
        \\// Input vertex attributes
        \\in vec3 vertexPosition;
        \\in vec2 vertexTexCoord;
        \\in vec3 vertexNormal;
        \\in vec4 vertexColor;
        \\
        \\// Input uniform values
        \\uniform mat4 mvp;
        \\uniform mat4 matModel;
        \\uniform mat4 matNormal;
        \\
        \\// Output vertex attributes (to fragment shader)
        \\out vec3 fragPosition;
        \\out vec2 fragTexCoord;
        \\out vec4 fragColor;
        \\out vec3 fragNormal;
        \\
        \\void main()
        \\{
        \\    // Send vertex attributes to fragment shader
        \\    fragPosition = vec3(matModel*vec4(vertexPosition, 1.0));
        \\    fragTexCoord = vertexTexCoord;
        \\    fragColor = vertexColor;
        \\    //fragNormal = normalize(vec3(matNormal*vec4(vertexNormal, 1.0)));
        \\    fragNormal = vertexNormal;
        \\
        \\    // Calculate final vertex position
        \\    gl_Position = mvp*vec4(vertexPosition, 1.0);
        \\}
    ,
        \\#version 330
        \\
        \\// Input vertex attributes (from vertex shader)
        \\in vec3 fragPosition;
        \\in vec2 fragTexCoord;
        \\in vec4 fragColor;
        \\in vec3 fragNormal;
        \\
        \\// Input uniform values
        \\uniform sampler2D texture0;
        \\uniform vec4 colDiffuse;
        \\uniform vec3 fogColor;
        \\uniform float depthMix;
        \\
        \\// Output fragment color
        \\out vec4 finalColor;
        \\
        \\// NOTE: Add here your custom variables
        \\
        \\void main()
        \\{
        \\    // Texel color fetching from texture sampler
        \\    vec4 texelColor = texture(texture0, fragTexCoord)*colDiffuse*fragColor;
        \\
        \\    float zNear = 0.001; // camera z near
        \\    float zFar = 3.0;  // camera z far
        \\    float z = gl_FragCoord.z;
        \\
        \\    // Linearize depth value
        \\    float depth = (2.0*zNear)/(zFar + zNear - z*(zFar - zNear));
        \\    
        \\    // Calculate final fragment color
        \\    finalColor = mix(vec4(fogColor, texelColor.a), texelColor, 1.0 - depth);
        \\}
    );
    defer ray.UnloadShader(shader);
    const depth_mix_loc = ray.GetShaderLocation(shader, "depthMix");
    const fog_color_loc = ray.GetShaderLocation(shader, "fogColor");

    var seed: u64 = 0;

    var mode: enum{mclone, house} = .mclone; // must be a runtime value

    var sample_interior = interior.generateRooms(seed);

    var mc = mclone.DisplayCache.init(std.heap.c_allocator);
    defer mc.deinit();

    var timer = try std.time.Timer.start();
    std.log.info("generating world…", .{});
    const xw = 4;
    const yw = 4;
    const zw = 4;
    for(mclone.range(xw)) |_, xun| {
        const x = @intCast(i32, xun) - (xw / 2);
        for(mclone.range(yw)) |_, yun| {
            const y = @intCast(i32, yun) - (yw / 2);
            for(mclone.range(zw)) |_, zun| {
                const z = @intCast(i32, zun) - (zw / 2);

                try mc.world.add(.{x, y, z}, try mclone.generateChunk(.{x, y, z}, mc.alloc));
            }
        }
    }
    _ = mc.world.setBlock(.{0, 0, 0}, .stone);
    std.log.info("generated in {d}ns", .{timer.lap()});
    try mc.update();
    std.log.info("generated meshes in {d}ns", .{timer.lap()});

    ray.SetCameraMode(camera, ray.CAMERA_CUSTOM);
    init_camera_first_person(&camera);

    ray.SetTargetFPS(60);

    var fog_color: ray.Color = .{.r = 209, .g = 247, .b = 255, .a = 255};

    while (!ray.WindowShouldClose()) {
        // ray.UpdateCamera(&camera);
        update_camera_first_person(&camera);

        var arena_allocator = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena_allocator.deinit();
        const arena = arena_allocator.allocator();

        if(ray.IsKeyPressed(ray.KEY_ONE)) {
            ray.EnableCursor();
        }
        if(ray.IsKeyPressed(ray.KEY_TWO)) {
            ray.DisableCursor();
        }
        if(ray.IsKeyPressed(ray.KEY_THREE)) {
            globalmode =! globalmode;
        }
        if(ray.IsKeyPressed(ray.KEY_FOUR)) {
            mode = @intToEnum(@TypeOf(mode), (@enumToInt(mode) +% 1) % std.meta.fields(@TypeOf(mode)).len);
        }
        if(ray.IsKeyPressed(ray.KEY_FIVE)) {
            linesmode = !linesmode;
        }

        const mouse_ray = ray.GetMouseRay(.{.x = window_width / 2, .y = window_height / 2}, camera);

        const rayhit = mclone.traceRay(
            mc.world,
            // fromRayvec(camera.position),
            // fromRayvec(camera.target),
            fromRayvec(mouse_ray.position),
            fromRayvec(mouse_ray.direction),
            mclone.BlockSize1 * 6,
        );

        switch(mode) {
            .mclone => {
                if(rayhit) |rh| {
                    if(ray.IsMouseButtonPressed(1)) {
                        _ = mc.world.setBlock(rh.block, .air);
                    }else if(ray.IsMouseButtonPressed(0)) { // no zero tick placing
                        _ = mc.world.setBlock(rh.block + rh.normal, .stone); // if this returns null, place failed
                    }
                }

                try mc.update();
            },
            .house => {
                if(ray.IsKeyPressed(ray.KEY_RIGHT)) {
                    seed +%= 1;
                    sample_interior = interior.generateRooms(seed);
                }else if(ray.IsKeyPressed(ray.KEY_LEFT)) {
                    seed -%= 1;
                    sample_interior = interior.generateRooms(seed);
                }
            },
        }

        ray.SetShaderValue(shader, depth_mix_loc, &@as(f32, 0.8), ray.SHADER_UNIFORM_FLOAT);
        ray.SetShaderValue(shader, fog_color_loc, &colorToVec3(fog_color), ray.SHADER_UNIFORM_VEC3);

        {
            ray.BeginDrawing();
            defer ray.EndDrawing();

            ray.ClearBackground(fog_color);
            {
                ray.BeginShaderMode(shader);
                defer ray.EndShaderMode();

                ray.BeginMode3D(camera);
                defer ray.EndMode3D();

                drawn_vertex_count = 0;
                drawn_batches_count = 0;
                switch(mode) {
                    .mclone => {
                        renderMclone(mc);
                        if(rayhit) |rh| {
                            const pos0 = mclone.toWorldPos(rh.block);
                            const pos1 = pos0 + mclone.BlockSize;

                            DrawQuad3D(
                                toRayvec(.{pos1[0], pos1[1] + 0.1, pos1[2]}),
                                toRayvec(.{pos1[0], pos1[1] + 0.1, pos0[2]}),
                                toRayvec(.{pos0[0], pos1[1] + 0.1, pos1[2]}),
                                toRayvec(.{pos0[0], pos1[1] + 0.1, pos0[2]}),
                                ray.RED,
                            );
                        }
                    },
                    .house => {
                        renderRooms3D(sample_interior);
                    },
                }
            }

            switch(mode) {
                .mclone => {
                    if(rayhit) |rh| {
                        _ = rh;
                        ray.DrawText(
                            try std.fmt.allocPrintZ(arena, "Ray hit: {d}", .{rh.block})
                        , 40, 200, 20, ray.BLACK);
                    }else{
                        ray.DrawText("ray no hit :(", 40, 200, 20, ray.BLACK);
                    }
                },
                .house => {
                    try renderRooms2D(sample_interior, arena);
                },
            }

            ray.DrawText("- Move with keys: W, A, S, D", 40, 40, 10, ray.DARKGRAY);
            ray.DrawText("- Mouse move to look around", 40, 60, 10, ray.DARKGRAY);
            ray.DrawText(try std.fmt.allocPrintZ(arena, "Seed: {d}", .{seed}), 40, 80, 20, ray.BLACK);
            ray.DrawText(try std.fmt.allocPrintZ(arena, "Tris: {d}×{d} + {d}", .{
                drawn_batches_count,
                max_vertex_count,
                drawn_vertex_count,
            }), 40, 100, 20, ray.BLACK);
            ray.DrawFPS(10, 10);

            ray.BeginBlendMode(ray.BLEND_SUBTRACT_COLORS);
            ray.DrawCircleLines(window_width / 2,  window_height / 2, 10, ray.WHITE);
            ray.DrawCircle(window_width / 2,  window_height / 2, 2, ray.WHITE);
            ray.EndBlendMode();
        }
    }
    //if (c_main() != 0) return error.Errored;
}

var globalmode = true;

fn toRayvec(pos: mclone.WorldPos) ray.Vector3 {
    return .{.x = pos[0], .y = pos[1], .z = pos[2]};
}
fn fromRayvec(rvec: ray.Vector3) mclone.WorldPos {
    return .{rvec.x, rvec.y, rvec.z};
}

fn renderMclone(dc: mclone.DisplayCache) void {
    var mesh_iter = dc.meshes.iterator();
    while(mesh_iter.next()) |kv| {
        renderMcloneChunk(kv.key_ptr.*, kv.value_ptr.*);
    }
}

fn renderMcloneChunk(offset_unscaled: mclone.BlockPos, mesh: mclone.ChunkMesh) void {
    const offset = mclone.toWorldPos(mclone.scaleOffset(offset_unscaled));

    for(mesh.quads) |quad| {
        DrawQuad3D(
            toRayvec(quad.points[0] + offset),
            toRayvec(quad.points[1] + offset),
            toRayvec(quad.points[2] + offset),
            toRayvec(quad.points[3] + offset),
            .{.r = quad.color[0], .g = quad.color[1], .b = quad.color[2], .a = 255},
        );
    }
}

fn renderRooms3D(sample_interior: interior.Interior) void {
    const room_scale = 5;

    for(sample_interior.rooms.slice()) |room| {
        if(room.outside) continue;

        const x0 = room.ul[0] * room_scale;
        const y0 = room.ul[1] * room_scale;
        const x1 = room.br[0] * room_scale;
        const y1 = room.br[1] * room_scale;
        DrawQuad3D(
            .{.x = x1, .y = 0, .z = y1},
            .{.x = x1, .y = 0, .z = y0},
            .{.x = x0, .y = 0, .z = y1},
            .{.x = x0, .y = 0, .z = y0},
            colorForInt(@enumToInt(room.kind)),
        );
    }

    for(sample_interior.walls.slice()) |wall| {
        switch(wall.door) {
            .none => {
                wall3d(wall.ul(), wall.br(), wall.xy, room_scale);
            },
            .door => {
                const door_offset: interior.Vec2f32 = switch(wall.xy) {
                    0 => .{interior.DOOR_SIZE / 2.0 - 0.1, 0},
                    1 => .{0, interior.DOOR_SIZE / 2.0 - 0.1},
                };
                const door_pos: interior.Vec2f32 = switch(wall.xy) {
                    0 => .{wall.door_up, wall.ul()[1]},
                    1 => .{wall.ul()[0], wall.door_up},
                };

                wall3d(wall.ul(), door_pos - door_offset, wall.xy, room_scale);
                wall3d(door_pos + door_offset, wall.br(), wall.xy, room_scale);
            },
            .no_wall => {},
        }
    }
}

fn renderRooms2D(sample_interior: interior.Interior, arena: std.mem.Allocator) !void {
    for(sample_interior.rooms.slice()) |room, i| {
        if(room.outside) continue;
        
        const xy = vec2f32toVec2cint(toMinimap(room.ul));
        const x0 = xy[0];
        const y0 = xy[1];

        const xy2 = vec2f32toVec2cint(toMinimap(room.br));

        ray.DrawRectangle(
            x0,
            y0,
            xy2[0] - x0,
            xy2[1] - y0,
            colorForInt(@enumToInt(room.kind)),
        );

        ray.DrawText(
            try std.fmt.allocPrintZ(arena, "{}, {s}", .{i, std.meta.tagName(room.kind)}),
            x0 + 2,
            y0 + 2,
            10,
            // colorForInt(i),
            ray.BLACK,
        );
    }

    for(sample_interior.walls.slice()) |wall| {
        switch(wall.door) {
            .none => {
                wall2d(wall.ul(), wall.br());
            },
            .door => {
                const door_offset: interior.Vec2f32 = switch(wall.xy) {
                    0 => .{interior.DOOR_SIZE / 2.0, 0},
                    1 => .{0, interior.DOOR_SIZE / 2.0},
                };
                const door_pos: interior.Vec2f32 = switch(wall.xy) {
                    0 => .{wall.door_up, wall.ul()[1]},
                    1 => .{wall.ul()[0], wall.door_up},
                };

                wall2d(wall.ul(), door_pos - door_offset);
                wall2d(door_pos + door_offset, wall.br());
            },
            .no_wall => {},
        }
    }
}