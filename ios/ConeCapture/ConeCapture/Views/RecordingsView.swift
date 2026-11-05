import SwiftUI

struct RecordingsView: View {
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var uploader: Uploader
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL

    var body: some View {
        List {
            if store.recordings.isEmpty {
                Text("No recordings yet.").foregroundStyle(.secondary)
            }
            ForEach(store.recordings) { rec in
                RecordingRow(rec: rec,
                             progress: store.progress[rec.id],
                             onRetry: { uploader.retry(rec.id) },
                             onOpen: { if let id = rec.jobID, let url = settings.jobPageURL(id: id) { openURL(url) } },
                             onDelete: { delete(rec) })
                    .swipeActions {
                        Button(role: .destructive) { delete(rec) } label: { Label("Delete", systemImage: "trash") }
                    }
            }
        }
        .navigationTitle("Recordings")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Upload all pending") { uploader.uploadAllPending() }
                    .disabled(!store.recordings.contains { ($0.status == .pending || $0.status == .failed) && !$0.mediaDeleted })
            }
            ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
        }
    }

    private func delete(_ rec: Recording) {
        if rec.status == .uploading { uploader.cancel(rec.id) }
        store.delete(rec.id)
    }
}

private struct RecordingRow: View {
    let rec: Recording
    let progress: Double?
    let onRetry: () -> Void
    let onOpen: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: rec.kind == .video ? "video.fill" : "photo.fill")
                VStack(alignment: .leading) {
                    Text(rec.name ?? rec.id).font(.headline)
                    Text("\(rec.id) \u{00B7} \(ByteCountFormatter.string(fromByteCount: rec.mediaBytes, countStyle: .file))\(rec.mediaDeleted ? " (media deleted)" : "")")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                statusBadge
            }
            if rec.status == .uploading {
                ProgressView(value: progress ?? 0)
                Text(String(format: "%.0f%%", (progress ?? 0) * 100)).font(.caption).foregroundStyle(.secondary)
            }
            if let msg = rec.errorMessage, rec.status != .uploaded {
                Text(msg).font(.caption).foregroundStyle(.red)
            }
            if let job = rec.jobID {
                HStack {
                    Text("Job \(job)").font(.caption.monospaced())
                    Spacer()
                    Button("Open in Safari", action: onOpen).font(.caption).buttonStyle(.bordered)
                }
            }
            HStack {
                if (rec.status == .failed || rec.status == .pending) && !rec.mediaDeleted {
                    Button(rec.status == .failed ? "Retry" : "Upload", action: onRetry).buttonStyle(.bordered)
                }
                Spacer()
                Button("Delete local", role: .destructive, action: onDelete).buttonStyle(.bordered)
            }
            .font(.caption)
        }
        .padding(.vertical, 4)
    }

    private var statusBadge: some View {
        let (text, color): (String, Color) = {
            switch rec.status {
            case .pending: return ("Pending", .orange)
            case .uploading: return ("Uploading", .blue)
            case .uploaded: return ("Uploaded", .green)
            case .failed: return ("Failed", .red)
            }
        }()
        return Text(text).font(.caption.bold()).foregroundStyle(color)
    }
}
