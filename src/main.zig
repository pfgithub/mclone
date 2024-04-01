const std = @import("std");
const core = @import("mach").core;
const gpu = core.gpu;
const mclone = @import("mclone.zig");

pub const App = @This();

title_timer: core.Timer,
pipeline: *gpu.RenderPipeline,

mc: mclone.DisplayCache,

pub fn init(app: *App) !void {
    core.setFrameRateLimit(60);
    try core.init(.{});

    const shader_module = core.device.createShaderModuleWGSL("shader.wgsl", @embedFile("shader.wgsl"));
    defer shader_module.release();

    // Fragment state
    const blend = gpu.BlendState{};
    const color_target = gpu.ColorTargetState{
        .format = core.descriptor.format,
        .blend = &blend,
        .write_mask = gpu.ColorWriteMaskFlags.all,
    };
    const fragment = gpu.FragmentState.init(.{
        .module = shader_module,
        .entry_point = "frag_main",
        .targets = &.{color_target},
    });
    const pipeline_descriptor = gpu.RenderPipeline.Descriptor{
        .fragment = &fragment,
        .vertex = gpu.VertexState{
            .module = shader_module,
            .entry_point = "vertex_main",
        },
    };
    const pipeline = core.device.createRenderPipeline(&pipeline_descriptor);

    app.* = .{
        .title_timer = try core.Timer.start(),
        .pipeline = pipeline,
        .mc = mclone.DisplayCache.init(core.allocator),
    };
    errdefer app.mc.deinit();

    var timer = try std.time.Timer.start();
    std.log.info("generating world…", .{});
    const xw = 4;
    const yw = 4;
    const zw = 4;
    for (0..xw) |xun| {
        const x = @as(i32, @intCast(xun)) - (xw / 2);
        for (0..yw) |yun| {
            const y = @as(i32, @intCast(yun)) - (yw / 2);
            for (0..zw) |zun| {
                const z = @as(i32, @intCast(zun)) - (zw / 2);

                try app.mc.world.add(.{ x, y, z }, try mclone.generateChunk(.{ x, y, z }, app.mc.alloc));
            }
        }
    }
    _ = app.mc.world.setBlock(.{ 0, 0, 0 }, .stone);
    std.log.info("generated in {d}ns", .{timer.lap()});
    try app.mc.update();
    std.log.info("generated meshes in {d}ns", .{timer.lap()});

    // notes:
    // - apparently we can async upload stuff to the gpu?
    // - so when new mesh data is available, we can send it off and then switch once it's fully sent
    // - also we can use one big buffer and render within it i think? instead of one draw call per chunk
    // not sure how that works
    // is that possible? draw() accepts vertex_count, instance_count, first_vertex, first_instance
    // but it takes multiple draw calls to draw from different places in the buffer, = one draw call
    // per chunk

    // > However, if each 3D model requires different vertex data (which is frequently the case),
    // > you can still render them using a single draw call. To do this, you would add another stream
    // > into your vertex data with glVertexAttribPointer (and glEnableVertexAttribArray). This extra
    // > stream would contain the index of the matrix within the uniform buffer that vertex should
    // > use when rendering - so each mesh within the VBO would have an identical index in the
    // > extra stream. The uniform buffer contains the same data as in the instancing setup.

    // is that helpful? it doesn't seem helpful

    // we could put all the mesh data in one thing and have some discarded vertices
    // is that fine to do?
}

pub fn deinit(app: *App) void {
    defer core.deinit();
    app.mc.deinit();
}

pub fn update(app: *App) !bool {
    var iter = core.pollEvents();
    while (iter.next()) |event| {
        switch (event) {
            .close => return true,
            else => {},
        }
    }

    const queue = core.queue;
    const back_buffer_view = core.swap_chain.getCurrentTextureView().?;
    const color_attachment = gpu.RenderPassColorAttachment{
        .view = back_buffer_view,
        .clear_value = std.mem.zeroes(gpu.Color),
        .load_op = .clear,
        .store_op = .store,
    };

    const encoder = core.device.createCommandEncoder(null);
    const render_pass_info = gpu.RenderPassDescriptor.init(.{
        .color_attachments = &.{color_attachment},
    });
    const pass = encoder.beginRenderPass(&render_pass_info);
    pass.setPipeline(app.pipeline);
    pass.draw(3, 1, 0, 0);
    pass.end();
    pass.release();

    var command = encoder.finish(null);
    encoder.release();

    queue.submit(&[_]*gpu.CommandBuffer{command});
    command.release();
    core.swap_chain.present();
    back_buffer_view.release();

    // update the window title every second
    if (app.title_timer.read() >= 1.0) {
        app.title_timer.reset();
        try core.printTitle("Triangle [ {d}fps ] [ Input {d}hz ]", .{
            core.frameRate(),
            core.inputRate(),
        });
    }

    return false;
}
