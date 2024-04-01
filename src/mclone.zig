// TODO replace f32 with f64 so it can fit all 32 bit ints inside it

const std = @import("std");

pub const Block = enum(u16) {
    air,
    stone,
    grass,

    pub fn solid(block: Block) bool {
        return switch (block) {
            .air => false,
            .stone => true,
            .grass => true,
        };
    }
    pub fn selectable(block: Block) bool {
        return switch (block) {
            .air => false,
            else => true,
        };
    }
};

pub const ChunkSizeX = 12;
pub const ChunkSizeY = 24;
pub const ChunkSizeZ = 16;
pub const ChunkSize = BlockPos{ ChunkSizeX, ChunkSizeY, ChunkSizeZ }; // ideally std.meta.Vector(3, comptime_int)
pub const BLOCKS_PER_CHUNK = @reduce(.Mul, ChunkSize);

pub const InChunkBlockPos = std.meta.Vector(3, i32);

pub const BlockPos = std.meta.Vector(3, i32);
pub const WorldPos = std.meta.Vector(3, f32);

pub const BlockSize = @splat(3, BlockSize1);
pub const BlockSize1: f32 = 1.5;

// ERROR; threadlocals were crashing. TODO
var c_cache: ?*const World = null;
var c_cache_pos: BlockPos = undefined;
var c_cache_chunk: ?*const Chunk = undefined;

pub const World = struct {
    chunks: std.AutoHashMap(BlockPos, Chunk),

    pub fn add(world: *World, offset_unscaled: BlockPos, chunk: Chunk) !void {
        c_cache = null;
        try world.chunks.putNoClobber(offset_unscaled, chunk);
    }

    pub fn chunkFromPosOpt(world: World, pos: BlockPos) ?Chunk {
        if (c_cache) |cc| if (cc == &world) {
            if (std.meta.eql(c_cache_pos, unscaleOffset(pos))) {
                return (c_cache_chunk orelse return null).*;
            }
        };
        const res = world.chunks.getEntry(unscaleOffset(pos));
        c_cache = &world;
        c_cache_pos = unscaleOffset(pos);
        c_cache_chunk = if (res) |r| r.value_ptr else null;
        return (c_cache_chunk orelse return null).*;
    }

    pub fn blockAt(world: World, pos: BlockPos) ?Block {
        const chunk = world.chunkFromPosOpt(pos) orelse return null;
        return chunk.blockAtRelative(pos - scaleOffset(unscaleOffset(pos)));
    }

    pub fn setBlock(world: *World, pos: BlockPos, block: Block) ?Block {
        const chunk = world.chunks.getEntry(unscaleOffset(pos)) orelse return null;
        return chunk.value_ptr.setBlock(
            Chunk.indexFromPosChunkRelative(pos - scaleOffset(unscaleOffset(pos))),
            block,
        );
    }
};

pub const DisplayCache = struct {
    world: World,
    meshes: std.AutoHashMap(BlockPos, ChunkMesh),
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) DisplayCache {
        return .{
            .world = .{
                .chunks = std.AutoHashMap(BlockPos, Chunk).init(alloc),
            },
            .meshes = std.AutoHashMap(BlockPos, ChunkMesh).init(alloc),
            .alloc = alloc,
        };
    }
    pub fn deinit(dc: *DisplayCache) void {
        var chunk_iter = dc.world.chunks.iterator();
        while (chunk_iter.next()) |kv| {
            kv.value_ptr.deinit(dc.alloc);
        }
        dc.world.chunks.deinit();

        var mesh_iter = dc.meshes.iterator();
        while (mesh_iter.next()) |kv| {
            kv.value_ptr.deinit(dc.alloc);
        }
        dc.meshes.deinit();
    }

    pub fn update(cache: *DisplayCache) !void {
        // var timer = try std.time.Timer.start();

        var max: usize = 7;

        var mesh_iter = cache.meshes.iterator();
        while (mesh_iter.next()) |kv| {
            if (cache.world.chunks.get(kv.key_ptr.*) != null) continue;
            // because I'm in an iterator I actually have an index I can remove
            // directly but there's no hm fn for that.
            kv.value_ptr.deinit(cache.alloc);
            std.debug.assert(cache.meshes.remove(kv.key_ptr.*));

            max -= 1;
            if (max == 0) return;
        }

        var iter = cache.world.chunks.iterator();
        while (iter.next()) |kv| {
            const gp = try cache.meshes.getOrPut(kv.key_ptr.*);

            var rw = false;
            var timer: ?std.time.Timer = null;
            if (!gp.found_existing) {
                rw = true;
            } else {
                if (!std.meta.eql(getChunkHashes(cache.world, kv.key_ptr.*), gp.value_ptr.chunk_hashes)) {
                    gp.value_ptr.deinit(cache.alloc);
                    rw = true;
                    timer = try std.time.Timer.start();
                }
            }

            if (rw) {
                gp.value_ptr.* = try generateMesh(cache.world, kv.key_ptr.*, kv.value_ptr.*, cache.alloc);

                if (timer) |*t| {
                    std.log.err("regenerated chunk in {d}ns -{}", .{ t.read(), kv.key_ptr.* });
                }

                max -= 1;
                if (max == 0) return;
            }
        }
    }
};

pub const Chunk = struct {
    hash: usize = 1,
    blocks: *[BLOCKS_PER_CHUNK]Block,

    pub fn update(chunk: *Chunk) void {
        chunk.hash += 1;
    }

    pub fn deinit(chunk: *Chunk, alloc: std.mem.Allocator) void {
        alloc.free(chunk.blocks);
    }

    pub const BlockIndex = std.math.IntFittingRange(0, BLOCKS_PER_CHUNK - 1);

    pub fn setBlock(chunk: *Chunk, index: BlockIndex, block: Block) Block {
        const prev = chunk.blocks[index];
        chunk.blocks[index] = block;
        chunk.update();
        return prev;
    }

    pub fn indexFromPosChunkRelative(pos: BlockPos) BlockIndex {
        return @intCast(
            BlockIndex,
            pos[x] * ChunkSizeY * ChunkSizeZ +
                pos[y] * ChunkSizeZ +
                pos[z],
        );
    }
    pub fn posFromIndexChunkRelative(index: BlockIndex) BlockPos {
        return BlockPos{
            (index / ChunkSizeY / ChunkSizeZ) % ChunkSizeX,
            (index / ChunkSizeZ) % ChunkSizeY,
            index % ChunkSizeZ,
        };
    }

    pub fn indexFromPosChunkRelativeOpt(pos: BlockPos) ?BlockIndex {
        if (@reduce(.Or, pos < BlockPos{ 0, 0, 0 }) or
            @reduce(.Or, pos >= ChunkSize))
        {
            return null;
        }
        return indexFromPosChunkRelative(pos);
    }

    pub fn blockAtRelative(chunk: Chunk, pos: BlockPos) ?Block {
        const index = indexFromPosChunkRelativeOpt(pos) orelse return null;
        return chunk.blocks[index];
    }
};
pub fn toWorldPos(pos: BlockPos) WorldPos {
    const xf = @intToFloat(f32, pos[x]);
    const yf = @intToFloat(f32, pos[y]);
    const zf = @intToFloat(f32, pos[z]);

    return WorldPos{ xf, yf, zf } * BlockSize;
}

// to generate a mesh, loop over the blocks in the chunk and draw all the needed triangles
// into an array. then, store that array.
//
// this must be done any time a chunk changes

const Quad = struct { points: [4]WorldPos, color: std.meta.Vector(3, u8) };
pub const ChunkMesh = struct {
    chunk_hashes: [7]usize,
    quads: []Quad,

    pub fn deinit(mesh: *ChunkMesh, alloc: std.mem.Allocator) void {
        alloc.free(mesh.quads);
    }

    // generate(mesh, chunk, alloc) // : [0]Quad{} .regenerate(chunk)
    // regenerate(mesh, chunk) // : arraylist from(quads) .resize(0)
    // deinit(mesh)
};

pub const Direction = struct {
    pub const x = BlockPos{ 1, 0, 0 };
    pub const y = BlockPos{ 0, 1, 0 };
    pub const z = BlockPos{ 0, 0, 1 };
};

fn solidInDir(world: World, pos: BlockPos, direction: BlockPos) bool {
    // todo this should require 5 chunks: the center + the 4 around it in order to simplify
    // meshes around chunk borders
    const block = world.blockAt(pos + direction) orelse return false;
    return block.solid();
}

const FaceDirection = struct {
    axis: enum(u3) {
        x = x,
        y = y,
        z = z,
    },
    sign: u1,
};

// ok
// [i % 4, i % 2, i % 1]

// pub fn face(direction: FaceDirection, pc: [2]WorldPos) [4]WorldPos {
//     const phase =

//     return .{
//         @shuffle(f32, pc[0], pc[1], .{@as(u32, x), ~@as(u32, y), ~@as(u32, z)}),
//         @shuffle(f32, pc[0], pc[1], .{@as(u32, x), ~@as(u32, y), @as(u32, z)}),
//         @shuffle(f32, pc[0], pc[1], .{@as(u32, x), @as(u32, y), ~@as(u32, z)}),
//         @shuffle(f32, pc[0], pc[1], .{@as(u32, x), @as(u32, y), @as(u32, z)}),
//         .{pc[0][x], pc[1][y], pc[1][z]},
//         .{pc[0][x], pc[1][y], pc[0][z]},
//         .{pc[0][x], pc[0][y], pc[1][z]},
//         .{pc[0][x], pc[0][y], pc[0][z]},
//     };
// }

pub fn generateMesh(world: World, chunk_offset_unscaled: BlockPos, chunk: Chunk, alloc: std.mem.Allocator) !ChunkMesh {
    var res = std.ArrayList(Quad).init(alloc);

    for (chunk.blocks) |block, i| {
        const color = switch (block) {
            .air => continue,
            .stone => std.meta.Vector(3, u8){ 235, 64, 52 },
            .grass => std.meta.Vector(3, u8){ 52, 235, 64 },
        };

        const pos_relative = Chunk.posFromIndexChunkRelative(@intCast(Chunk.BlockIndex, i));
        const pos = pos_relative + scaleOffset(chunk_offset_unscaled);
        const pos0 = toWorldPos(pos_relative);
        const pos1 = pos0 + BlockSize;

        const pc = [_]WorldPos{ pos0, pos1 };

        // it should be possible to do this with a loop somehow

        // ok so it's just a pattern

        if (!solidInDir(world, pos, -Direction.x)) try res.append(.{
            .points = .{
                .{ pc[0][x], pc[1][y], pc[1][z] },
                .{ pc[0][x], pc[1][y], pc[0][z] },
                .{ pc[0][x], pc[0][y], pc[1][z] },
                .{ pc[0][x], pc[0][y], pc[0][z] },
            },
            .color = color,
        });
        if (!solidInDir(world, pos, Direction.x)) try res.append(.{
            .points = .{
                .{ pc[1][x], pc[1][y], pc[0][z] },
                .{ pc[1][x], pc[1][y], pc[1][z] },
                .{ pc[1][x], pc[0][y], pc[0][z] },
                .{ pc[1][x], pc[0][y], pc[1][z] },
            },
            .color = color,
        });
        if (!solidInDir(world, pos, -Direction.y)) try res.append(.{
            .points = .{
                .{ pc[1][x], pc[0][y], pc[1][z] },
                .{ pc[0][x], pc[0][y], pc[1][z] },
                .{ pc[1][x], pc[0][y], pc[0][z] },
                .{ pc[0][x], pc[0][y], pc[0][z] },
            },
            .color = color,
        });
        if (!solidInDir(world, pos, Direction.y)) try res.append(.{
            .points = .{
                .{ pc[0][x], pc[1][y], pc[1][z] },
                .{ pc[1][x], pc[1][y], pc[1][z] },
                .{ pc[0][x], pc[1][y], pc[0][z] },
                .{ pc[1][x], pc[1][y], pc[0][z] },
            },
            .color = color,
        });
        if (!solidInDir(world, pos, -Direction.z)) try res.append(.{
            .points = .{
                .{ pc[1][x], pc[1][y], pc[0][z] },
                .{ pc[1][x], pc[0][y], pc[0][z] },
                .{ pc[0][x], pc[1][y], pc[0][z] },
                .{ pc[0][x], pc[0][y], pc[0][z] },
            },
            .color = color,
        });
        if (!solidInDir(world, pos, Direction.z)) try res.append(.{
            .points = .{
                .{ pc[1][x], pc[0][y], pc[1][z] },
                .{ pc[1][x], pc[1][y], pc[1][z] },
                .{ pc[0][x], pc[0][y], pc[1][z] },
                .{ pc[0][x], pc[1][y], pc[1][z] },
            },
            .color = color,
        });
    }

    return ChunkMesh{
        .quads = res.toOwnedSlice(),
        .chunk_hashes = getChunkHashes(world, chunk_offset_unscaled),
    };
}

pub fn getChunkHashes(world: World, offset_unscaled: BlockPos) [7]usize {
    return .{
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled))) |v| v.hash else 0,
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled + Direction.x))) |v| v.hash else 0,
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled - Direction.x))) |v| v.hash else 0,
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled + Direction.y))) |v| v.hash else 0,
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled - Direction.y))) |v| v.hash else 0,
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled + Direction.z))) |v| v.hash else 0,
        if (world.chunkFromPosOpt(scaleOffset(offset_unscaled - Direction.z))) |v| v.hash else 0,
    };
}

pub fn scaleOffset(offset_unscaled: BlockPos) BlockPos {
    return offset_unscaled * ChunkSize;
}
pub fn unscaleOffset(offset_scaled: BlockPos) BlockPos {
    return @divFloor(offset_scaled, ChunkSize);
}

pub fn generateChunk(offset_unscaled: BlockPos, alloc: std.mem.Allocator) !Chunk {
    const blocks = try alloc.create([BLOCKS_PER_CHUNK]Block);
    errdefer alloc.destroy(blocks);

    var res: Chunk = .{
        .blocks = blocks,
    };

    for (range(ChunkSizeX)) |_, xp| {
        for (range(ChunkSizeY)) |_, yp| {
            for (range(ChunkSizeZ)) |_, zp| {
                const pos_relative = BlockPos{ @intCast(i32, xp), @intCast(i32, yp), @intCast(i32, zp) };
                const pos = pos_relative + scaleOffset(offset_unscaled);

                const block: Block = switch (pos[y]) {
                    0...std.math.maxInt(i32) => .air,
                    -1 => .grass,
                    std.math.minInt(i32)...-2 => .stone,
                };

                _ = res.setBlock(Chunk.indexFromPosChunkRelative(pos_relative), block);
            }
        }
    }

    return res;
}

pub fn range(max: usize) []const void {
    return @as([]const void, &[_]void{}).ptr[0..max];
}

pub fn normalize(dir: WorldPos) WorldPos {
    const len = std.math.sqrt(@reduce(.Add, dir * dir)); // sqrt(...dirv);
    if (len == 0) unreachable; // attempt to normalize directionless vector
    return dir / @splat(3, len);
}

pub fn traceRay(
    world: World,
    world_pos: WorldPos,
    direction: WorldPos, // unnormalized
    max_d: f32,
) ?RayHit {
    const Data = struct {
        world: *const World,
    };
    const data = Data{ .world = &world };

    const p = world_pos / BlockSize;
    const d = normalize(direction / BlockSize);

    const getVoxel = struct {
        fn getVoxel(data_: usize, pos: BlockPos) bool {
            const theworld = @intToPtr(*const Data, data_).world;

            const block = theworld.blockAt(pos) orelse return false;
            return block.selectable();
        }
    }.getVoxel;

    const res = traceRay_impl(
        @ptrToInt(&data),
        getVoxel,
        p,
        d,
        max_d / BlockSize1,
    ) orelse return null;

    return RayHit{
        .block = res.block,
        .world = res.world_unscaled * BlockSize,
        .normal = res.normal,
    };
}

const RayHit = struct {
    block: BlockPos,
    world: WorldPos,
    normal: BlockPos,
};

const RayHitRaw = struct {
    block: BlockPos,
    world_unscaled: WorldPos, // make sure to * BlockSize
    normal: BlockPos,
};

fn i32x3_to_f32x3(ints: std.meta.Vector(3, i32)) std.meta.Vector(3, f32) {
    return .{
        @intToFloat(f32, ints[x]),
        @intToFloat(f32, ints[y]),
        @intToFloat(f32, ints[z]),
    };
}

fn f32x3_to_i32x3(floats: std.meta.Vector(3, f32)) std.meta.Vector(3, i32) {
    return .{
        @floatToInt(i32, floats[x]),
        @floatToInt(i32, floats[y]),
        @floatToInt(i32, floats[z]),
    };
}

fn traceRay_impl(
    data: usize,
    getVoxel: fn (data: usize, pos: BlockPos) bool,
    p: WorldPos,
    d: WorldPos,
    max_d: f32,
) ?RayHitRaw {
    // https://github.com/fenomas/fast-voxel-raycast/blob/master/index.js, MIT licensed
    // consider raycast vector to be parametrized by t
    //   vec = [px,py,pz] + t * [dx,dy,dz]

    // algo below is as described by this paper:
    // http://www.cse.chalmers.se/edu/year/2010/course/TDA361/grid.pdf
    var t: f32 = 0.0;
    var i = f32x3_to_i32x3(@floor(p));

    var step = @select(
        i2,
        d > @splat(3, @as(f32, 0)),
        @splat(3, @as(i2, 1)),
        @splat(3, @as(i2, -1)),
    );

    var t_delta = @fabs(@splat(3, @as(f32, 1)) / d);

    var dist = @select(
        f32,
        step > @splat(3, @as(i2, 0)),
        i32x3_to_f32x3(i + @splat(3, @as(i32, 1))) - p,
        p - i32x3_to_f32x3(i),
    );

    var t_max = @select(
        f32,
        t_delta < @splat(3, std.math.inf(f32)),
        t_delta * dist,
        @splat(3, std.math.inf(f32)),
    );

    var stepped_dir: enum(u2) { x = x, y = y, z = z, na } = .na;

    while (t <= max_d) {
        if (stepped_dir != .na and getVoxel(data, i)) {
            return RayHitRaw{
                .block = i,
                .world_unscaled = p + @splat(3, t) * d,
                .normal = switch (stepped_dir) {
                    .na => unreachable,
                    .x => .{ -step[x], 0, 0 },
                    .y => .{ 0, -step[y], 0 },
                    .z => .{ 0, 0, -step[z] },
                },
            };
        }

        // advance t to next nearest voxel boundary

        // so I compressed this code but I'm not entirely sure
        // what it does. it's supposed to find the nearest voxel
        // boundary but I'm not entirely sure how the original code
        // does that and I'm definitely not sure how this code does
        // that. the original code might have been better than this.
        inline for (.{ x, y, z }) |dir| {
            const cond = switch (dir) {
                x => t_max[x] < t_max[y] and t_max[x] < t_max[z],
                y => t_max[y] < t_max[z],
                z => true,
                else => unreachable,
            };

            if (cond) {
                i[dir] += step[dir];
                t = t_max[dir];
                t_max[dir] += t_delta[dir];
                stepped_dir = @intToEnum(@TypeOf(stepped_dir), dir);
                break;
            }
        }
    }

    // no voxel hit found
    return null;
}

pub const x = 0;
pub const y = 1;
pub const z = 2;

// lighting
// mc-style lighting is simple
// - sky lighting: store a big 2d grid with the highest block at any coordinate.
//   also works for rain.
// - block lighting: same as mc
//
// if I wanted to do more fun lighting I think it would be a lot more complicated
// unfortunately. I can do boring shader lighting where you use a camera and the depth
// map and stuff.
//
//
// - time of day based lighting
//   when you place a block:
//   - do a bunch of raycasts to find affected blocks and set their time of day
//     value eg a u8 to the value they are. a u8 only allows 8 times of day. it
//     needs 1 bit per time of day. huh. maybe u32
//   - for a block to recalculate its time of day value it just has to do like
//     32 raycasts to see what/if anything is obstructing it from seeing the sky
//   - when a block change edits another block's time of day value, it can just
//     edit the bits it changes directly. no need to do 1024 checks for updating
//     one block
//   - we can smoothly transition between different times of day even if there
//     are only eg 32
//   - don't use `u32` directly, instead use like BitArray(32) or something.
//     it would be a zero-cost abstraction so literally the same as u32 but more
//     extensible
//
//
// ok the more difficult one
//
// - user-placed lighting
//   - this is easy if there's a light count limit of like 32 lights but that's
//     obviously not possible.
//   - when a block changes, it needs to know the coords of all the lights acting
//     upon it so light values can be recalculated
//   - alternatively, we can use f32s and be like mc lighting but circular
//     - this doesn't allow for things blocking other things
//
// ok
// - for any given voxel, store the closest light source
// - when a block is placed, we now need to do a lot of light updates. like basically
//   we need to make some extruded polygon shape and tell all those blocks to update
//   their light
// - when we delete a light, we have to flood fill search all blocks which have this
//   marked as their 'closest light', then starting at the edges and going in we have
//   to recalculate the closest lights using info from the blocks around
//
// - will that work?
//   - the issue is when placing a block.
//   - breaking a block is easy - for the 26 blocks around it, recalculate nearest lights
//     in a loop until all related things have been recalculated. use a resonable light
//     length limit to make sure this can only update like eg 4,000 blocks at most
//   - placing a block?
// - ok I messed around with this https://ricktu288.github.io/ray-optics/simulator/ for a
//   bit I don't think the above described method can possibly work at all
// - the main issue is that sometimes rays have to pass through other rays and there's
//   no way to figure out which light to use by just looking at the neighbors in that
//   case
// - to fix that, we'd have to store eg 32 lights per block and then we run into light
//   maximum issues that I wanted to avoid

// the plan:
// - lighting:
//   - BitArray(32) sunlight so it can cast shadows and stuff. interpolate between
//     things for more smoothness rather than jumping one bit at a time
//     - note that this does not allow for opacity. opacity would require [32]f16 which
//       is more than I want. although it might be possible to only store sun light values
//       for blocks that are exposed to the surface, so that might be possible.
//   - f16 block lighting - it doesn't cast shadows very well but there's not much
//     we can do about that
// - occlusion culling: https://tomcc.github.io/2014/08/31/visibility-1.html
// - lod: https://0fps.net/2018/03/03/a-level-of-detail-method-for-blocky-voxels/

// - lighting
//   - skylight v2
//     - rather than associating this info with the block, just keep a bunch of
//       maps for the coords of the highest block along the axis. this is wayy
//       cheaper by data usage (doesn't require a 32 bit int per xyz coord, just per
//       xz coordinate)
//     - fun!
